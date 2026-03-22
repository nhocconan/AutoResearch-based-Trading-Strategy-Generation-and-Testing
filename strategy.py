#!/usr/bin/env python3
"""
Experiment #387: 1h Connors RSI + 4h HMA Trend + BB Width Regime + ATR Stop

Hypothesis: After analyzing 386 failed experiments, the pattern is clear:
- Pure trend-following fails in 2022 crash and 2025 bear market
- Pure mean-reversion fails in strong trends
- The winning approach (#376) used regime detection + adaptive entries

This strategy combines:
1. CONNORS RSI (CRSI): Proven 75% win rate in literature for mean-reversion
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long when CRSI < 15, Short when CRSI > 85
   - Much more responsive than standard RSI(14)

2. 4h HMA(21) TREND FILTER: Not too slow (like 1d), not too fast (like 1h)
   - Only long when price > 4h HMA (bullish bias)
   - Only short when price < 4h HMA (bearish bias)
   - HMA has less lag than EMA for trend detection

3. BOLLINGER BAND WIDTH REGIME: Detects squeeze vs expansion
   - BB Width < 20th percentile = squeeze (breakout coming, reduce mean-reversion)
   - BB Width > 80th percentile = expansion (mean-reversion works well)
   - Adaptive position sizing based on regime

4. VOLUME CONFIRMATION: Filter false signals
   - Volume > 0.8 * SMA(volume, 20) to confirm genuine moves

5. ATR TRAILING STOP (2.0x): Protect from crashes
   - Signal → 0 when price moves 2.0*ATR against position

6. ASYMMETRIC EXITS: Easier to enter, harder to exit
   - Enter at CRSI < 15 / > 85
   - Exit at CRSI < 40 / > 60 (give room for trend to develop)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing

Why this should beat current best (Sharpe=0.676):
- CRSI is more responsive than standard RSI for 1h timeframe
- 4h HMA is better aligned with 1h entries than 1d HMA
- BB Width regime filter adapts to volatility conditions
- Asymmetric exits let winners run while cutting losers fast
- Should generate 40-80 trades/year per symbol (enough for stats)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_bbwidth_vol_atr_v1"
timeframe = "1h"
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
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI for quick mean-reversion signals
    RSI_Streak(2): RSI of consecutive up/down streaks
    PercentRank(100): Percentile rank of 1-day price change
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI of Streak
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi_vals = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi_vals.values
    
    # PercentRank of 1-period returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_bb_width(high, low, close, period=20, std_dev=2.0):
    """
    Calculate Bollinger Band Width as regime indicator.
    BB Width = (Upper Band - Lower Band) / Middle Band
    Low width = squeeze (breakout coming)
    High width = expansion (mean-reversion works)
    """
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bb_width = (upper - lower) / (middle + 1e-10)
    return bb_width.values

def calculate_bb_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for regime detection."""
    n = len(bb_width)
    bb_percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bb_width[i-lookback+1:i+1]
        current = bb_width[i]
        if not np.isnan(current):
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                bb_percentile[i] = np.sum(valid_window < current) / len(valid_window) * 100
    
    return bb_percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_width = calculate_bb_width(high, low, close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_width, 100)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_NORMAL = 0.25
    SIZE_EXPANSION = 0.30  # Higher size when BB width is high (mean-reversion works better)
    
    # Track position state for stoploss and asymmetric exits
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
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
        
        if np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # Strong mean-reversion long signal
        crsi_overbought = crsi[i] > 85  # Strong mean-reversion short signal
        
        # Asymmetric exit thresholds (easier to enter, harder to exit)
        crsi_exit_long = crsi[i] > 60  # Exit long when CRSI rises
        crsi_exit_short = crsi[i] < 40  # Exit short when CRSI falls
        
        # === BB WIDTH REGIME ===
        bb_squeeze = bb_percentile[i] < 20  # Low volatility, breakout coming
        bb_expansion = bb_percentile[i] > 80  # High volatility, mean-reversion works
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * volume_sma[i] if not np.isnan(volume_sma[i]) else True
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Determine position size based on regime
        current_size = SIZE_EXPANSION if bb_expansion else SIZE_NORMAL
        
        # LONG ENTRY: CRSI oversold + bull trend + volume ok
        # Reduce position size during squeeze (breakout risk)
        if crsi_oversold and bull_trend_4h and volume_ok:
            if bb_squeeze:
                new_signal = current_size * 0.7  # Reduce size during squeeze
            else:
                new_signal = current_size
        
        # SHORT ENTRY: CRSI overbought + bear trend + volume ok
        elif crsi_overbought and bear_trend_4h and volume_ok:
            if bb_squeeze:
                new_signal = -current_size * 0.7  # Reduce size during squeeze
            else:
                new_signal = -current_size
        
        # === ASYMMETRIC EXIT LOGIC ===
        # Exit long position when CRSI rises above 60 (give room for trend)
        if in_position and position_side > 0 and crsi_exit_long:
            new_signal = 0.0
        
        # Exit short position when CRSI falls below 40 (give room for trend)
        if in_position and position_side < 0 and crsi_exit_short:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and new_signal != 0.0:
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