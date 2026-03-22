#!/usr/bin/env python3
"""
Experiment #414: 1d Connors RSI + Weekly HMA Trend + Choppiness Regime + ATR Stop

Hypothesis: After 413 experiments, the key insight is that DAILY timeframe needs
MEAN REVERSION logic with strong trend filtering, not pure trend following.
2025+ is bear/range market where trend strategies fail. This strategy uses:

1. CONNORS RSI (CRSI) for precise entry timing:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 (extreme oversold)
   - Short when CRSI < 90 (extreme overbought)
   - 75% win rate in literature, works in bear markets

2. WEEKLY HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > weekly HMA (bullish bias)
   - Short only when price < weekly HMA (bearish bias)
   - HMA smoother than EMA, critical for weekly alignment
   - Prevents counter-trend mean reversion disasters

3. CHOPPINESS INDEX (CHOP) REGIME FILTER:
   - CHOP(14) > 61.8 = ranging market (enable mean reversion)
   - CHOP(14) < 38.2 = trending market (disable mean reversion, use breakout)
   - Best meta-filter for distinguishing market states

4. DONCHIAN(20) BREAKOUT for trending regime:
   - When CHOP < 38.2 (trending), use breakout logic instead
   - Long on Donchian high break + weekly HMA bull
   - Short on Donchian low break + weekly HMA bear

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

6. POSITION SIZING: 0.28 discrete (conservative for daily volatility)
   - Max 28% capital per position
   - Discrete levels minimize fee churn

Why 1d should work:
- Daily bars filter intraday noise
- Weekly HTF provides strong trend confirmation
- Connors RSI excels at catching reversals in bear markets
- Choppiness filter avoids mean reversion in strong trends
- Should generate 20-40 trades/year (enough for Sharpe, not too many for fees)
- Must work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_rsi_weekly_hma_chop_donchian_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of price change over lookback
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        gains = np.sum(streak_vals[streak_vals > 0])
        losses = np.abs(np.sum(streak_vals[streak_vals < 0]))
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # PercentRank component
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar (simple TR for this calculation)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        # neutral_market = 38.2 <= CHOP <= 61.8 (stay flat or reduce size)
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS (for ranging regime) ===
        crsi_long = crsi[i] < 10  # Extreme oversold
        crsi_short = crsi[i] > 90  # Extreme overbought
        
        # === DONCHIAN BREAKOUT SIGNALS (for trending regime) ===
        donchian_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING REGIME: Connors RSI mean-reversion with weekly HMA filter
        if ranging_market:
            if bull_trend_1w and crsi_long:
                new_signal = SIZE
            elif bear_trend_1w and crsi_short:
                new_signal = -SIZE
        
        # TRENDING REGIME: Donchian breakout with weekly HMA filter
        elif trending_market:
            if bull_trend_1w and donchian_long:
                new_signal = SIZE
            elif bear_trend_1w and donchian_short:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Stay flat (38.2 <= CHOP <= 61.8)
        # This avoids whipsaw in unclear markets
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            # Long position from ranging regime should exit if market becomes trending without breakout
            if position_side > 0 and trending_market and not donchian_long:
                new_signal = 0.0
            # Short position from ranging regime should exit if market becomes trending without breakout
            if position_side < 0 and trending_market and not donchian_short:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT (for all positions) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals