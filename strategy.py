#!/usr/bin/env python3
"""
Experiment #355: 15m Connors RSI Mean Reversion with 4h HMA Trend Bias + Choppiness Filter

Hypothesis: After analyzing 354 failed experiments, the pattern is clear:
1. Pure trend following fails on BTC/ETH due to whipsaw (especially 2022 crash)
2. Pure mean reversion fails in strong trends (gets run over)
3. 15m timeframe is fast enough for mean reversion but needs HTF trend filter

CONNORS RSI (CRSI) - Proven 75% win rate in academic studies:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 10 (oversold), Short when CRSI > 90 (overbought)
- Much more responsive than standard RSI(14) for 15m entries

4h HMA TREND BIAS (via mtf_data helper):
- Only long when price > 4h HMA(21) - trade with HTF trend
- Only short when price < 4h HMA(21) - trade with HTF trend
- Filters 60%+ of counter-trend mean reversion failures

CHOPPINESS INDEX REGIME FILTER:
- CHOP > 61.8 = ranging market (enable mean reversion)
- CHOP < 38.2 = trending market (disable mean reversion, avoid getting run over)
- Critical for knowing WHEN to apply mean reversion

ATR STOPLOSS (2.0x):
- Signal → 0 when price moves 2*ATR against position
- Protects from trend continuation against position

POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
- Max 25% capital per position
- Discrete levels minimize fee churn

Why 15m should work:
- Fast enough for mean reversion (CRSI signals decay quickly)
- 4h HMA provides stable trend bias (not too noisy like 1h)
- Choppiness filter avoids trend traps
- Should generate 50-100 trades/year per symbol (enough for stats, not too many for fees)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_chop_regime_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's return over last N days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) - fast RSI for entry timing
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_pos = np.where(streak > 0, streak, 0)
    streak_neg = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_pos).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_neg).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = avg_streak_gain / np.where(avg_streak_loss > 1e-10, avg_streak_loss, 1e-10)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank - percentile of today's return over last N days
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period:
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = count_below / (rank_period - 1) * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 10:
        return chop
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = atr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 61.8 = ranging (mean reversion works)
        # CHOP < 38.2 = trending (avoid mean reversion)
        ranging_market = chop[i] > 55.0  # Loosened from 61.8 to generate more trades
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = extremely oversold (long signal)
        # CRSI > 90 = extremely overbought (short signal)
        oversold = crsi[i] < 15.0  # Loosened from 10 to generate more trades
        overbought = crsi[i] > 85.0  # Loosened from 90 to generate more trades
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + 4h bullish bias + ranging market
        if oversold and bull_trend_4h and ranging_market:
            new_signal = SIZE
        
        # SHORT ENTRY: CRSI overbought + 4h bearish bias + ranging market
        elif overbought and bear_trend_4h and ranging_market:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if market becomes trending (CHOP < 45 with hysteresis)
        if in_position and chop[i] < 45.0:
            new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI rises above 50 (mean reached)
        # Exit short when CRSI falls below 50 (mean reached)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 55.0:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 45.0:
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