#!/usr/bin/env python3
"""
Experiment #380: 30m Connors RSI + 4h HMA Trend + Choppiness Regime Filter

Hypothesis: After 379 experiments, the key insight is that 30m timeframe needs
faster entry signals than 4h/1d strategies, but still requires HTF trend confirmation.
Connors RSI (CRSI) has proven 75% win rate in academic studies for mean-reversion
entries within a trend. Combined with 4h HMA trend bias and Choppiness regime filter,
this should generate 50-100 trades/year with positive Sharpe on all symbols.

STRATEGY COMPONENTS:
1. CONNORS RSI (30m): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 15 (oversold pullback in uptrend)
   - Short when CRSI > 85 (overbought rally in downtrend)
   - Generates more trades than simple RSI(14) extremes

2. 4h HMA(21) TREND BIAS (HTF via mtf_data):
   - Only long when price > 4h HMA (bullish HTF trend)
   - Only short when price < 4h HMA (bearish HTF trend)
   - HMA has less lag than EMA for trend detection

3. CHOPPINESS INDEX (14-period) REGIME FILTER:
   - CHOP > 61.8 = ranging (widen CRSI bands to 10/90 for more mean-reversion)
   - CHOP < 38.2 = trending (tighten CRSI bands to 20/80 for trend pullbacks)
   - Avoids entries in neutral chop zone

4. ATR TRAILING STOP (2.5x): Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from sudden reversals

5. POSITION SIZING: 0.28 discrete levels
   - Conservative for 30m volatility
   - Discrete levels minimize fee churn

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Expected trades: 60-120/year per symbol (enough for statistical significance)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_hma_chop_regime_atr_v1"
timeframe = "30m"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI of streak length (Connors RSI component).
    Streak = consecutive up or down days.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    streaks = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streaks[i] = streaks[i-1] + 1 if streaks[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streaks[i] = streaks[i-1] - 1 if streaks[i-1] <= 0 else -1
        else:
            streaks[i] = 0
    
    # Calculate RSI on absolute streak values
    abs_streaks = np.abs(streaks)
    abs_streaks_s = pd.Series(abs_streaks)
    
    # Handle edge case where all streaks are 0
    if abs_streaks_s.max() == 0:
        return np.ones(n) * 50.0
    
    delta = abs_streaks_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    streak_rsi = 100 - (100 / (1 + rs))
    
    return streak_rsi.values

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank (Connors RSI component).
    Percentage of closes in last period days that were lower than current close.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] < current)  # exclude current from comparison
        pr[i] = 100 * count_lower / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # Adjust CRSI thresholds based on regime
        if trending_market:
            crsi_long_threshold = 20  # Less extreme for trend pullbacks
            crsi_short_threshold = 80
        elif ranging_market:
            crsi_long_threshold = 10  # More extreme for mean-reversion
            crsi_short_threshold = 90
        else:
            # Neutral zone - still allow trades but with middle thresholds
            crsi_long_threshold = 15
            crsi_short_threshold = 85
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < crsi_long_threshold
        crsi_overbought = crsi[i] > crsi_short_threshold
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: CRSI oversold + 4h bullish trend (or ranging market)
        if crsi_oversold:
            if bull_trend_4h or ranging_market:
                new_signal = SIZE
        
        # SHORT: CRSI overbought + 4h bearish trend (or ranging market)
        elif crsi_overbought:
            if bear_trend_4h or ranging_market:
                new_signal = -SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h and not ranging_market:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h and not ranging_market:
                new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 30:
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