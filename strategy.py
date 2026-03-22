#!/usr/bin/env python3
"""
Experiment #109: 15m Regime-Adaptive Strategy with 4h HMA Trend Filter

Hypothesis: Building on lessons from 95+ failed strategies, this implements a
regime-adaptive approach that switches between mean-reversion and trend-following
based on market conditions. Key innovations:

1. CONNORS RSI (CRSI): Combines RSI(3) + RSI_Streak(2) + PercentRank(100) for
   superior mean-reversion signals. Entry at CRSI<15 (long) or CRSI>85 (short).
   
2. CHOPPINESS INDEX (CHOP): Detects regime - CHOP>61.8 = range (mean-revert),
   CHOP<38.2 = trend (trend-follow). This meta-filter prevents trend strategies
   from failing in choppy markets and vice-versa.

3. 4h HMA(21) Trend Bias: Higher timeframe filter ensures we only take longs
   when 4h trend is bullish, shorts when 4h trend is bearish. Prevents
   counter-trend trades that destroyed strategies in 2022 crash.

4. ATR Trailing Stop (2.5*ATR): Protects capital during adverse moves. Critical
   for surviving BTC's 77% crash in 2022.

5. Discrete Position Sizing (0.20/0.30): Minimizes fee churn while maintaining
   exposure. Max 30% capital per position.

Why 15m timeframe: Faster signals than 4h/1d strategies, more trade opportunities
while still capturing meaningful moves. 15m has sufficient liquidity on Binance
for BTC/ETH/SOL perpetuals.

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_regime_4h_hma_atr_v1"
timeframe = "15m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): RSI of consecutive up/down days
    - PercentRank(100): Where current price ranks in last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (mean-reversion favorable)
    - CHOP < 38.2: Market is trending (trend-following favorable)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # Calculate ATR for each bar
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
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Need enough data for CRSI rank_period=100 + CHOP period=14
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (mean-reversion favorable)
        # CHOP < 38.2 = trending (trend-following favorable)
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (Choppy Market) ===
        if is_choppy:
            # Long: CRSI < 15 (oversold) + 4h trend bullish
            if crsi[i] < 15 and bull_trend_4h:
                new_signal = SIZE_STRONG
            # Long: CRSI < 20 (moderate oversold) + 4h trend bullish
            elif crsi[i] < 20 and bull_trend_4h:
                new_signal = SIZE_BASE
            
            # Short: CRSI > 85 (overbought) + 4h trend bearish
            elif crsi[i] > 85 and bear_trend_4h:
                new_signal = -SIZE_STRONG
            # Short: CRSI > 80 (moderate overbought) + 4h trend bearish
            elif crsi[i] > 80 and bear_trend_4h:
                new_signal = -SIZE_BASE
        
        # === TREND FOLLOWING MODE (Trending Market) ===
        elif is_trending:
            # Long: 4h bullish + pullback to EMA21 + CRSI not overbought
            if bull_trend_4h and close[i] > ema_21[i] and crsi[i] < 70:
                new_signal = SIZE_STRONG
            # Long: 4h bullish + price above EMA21 (momentum)
            elif bull_trend_4h and close[i] > ema_21[i] and close[i] > hma_4h_aligned[i]:
                new_signal = SIZE_BASE
            
            # Short: 4h bearish + pullback to EMA21 + CRSI not oversold
            elif bear_trend_4h and close[i] < ema_21[i] and crsi[i] > 30:
                new_signal = -SIZE_STRONG
            # Short: 4h bearish + price below EMA21 (momentum)
            elif bear_trend_4h and close[i] < ema_21[i] and close[i] < hma_4h_aligned[i]:
                new_signal = -SIZE_BASE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        # Use lighter positions, require stronger signals
        else:
            # Long: Very oversold + 4h bullish
            if crsi[i] < 10 and bull_trend_4h:
                new_signal = SIZE_BASE
            # Short: Very overbought + 4h bearish
            elif crsi[i] > 90 and bear_trend_4h:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals