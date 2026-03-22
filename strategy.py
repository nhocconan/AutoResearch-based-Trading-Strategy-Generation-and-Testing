#!/usr/bin/env python3
"""
Experiment #396: 1d Weekly HMA Trend + Daily RSI Pullback + BB Squeeze Breakout

Hypothesis: After 395 failed experiments, the key insight for 1d timeframe is:
1. Weekly HMA(21) provides stable long-term trend bias (weekly closes matter)
2. Daily RSI(14) pullbacks in trend direction have high win rate (~65%)
3. Bollinger Band squeeze (width < 20th percentile) precedes explosive moves
4. ADX(14) > 20 filters out dead markets (avoid chop)
5. 1d naturally produces fewer but higher quality signals (20-40/year)

Why this should work on 1d:
- Daily bars have less noise than intraday
- Weekly trend filter avoids counter-trend trades in strong moves
- RSI pullback entries (not extremes) catch continuations
- BB squeeze breakout captures volatility expansion
- Should generate 80-160 trades over 4y train (well above 10 minimum)
- Conservative sizing (0.30) protects from 2022-style crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_rsi_pullback_bb_squeeze_adx_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx[period*2:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period*2:]
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma * 100.0
    return upper, lower, bandwidth

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_percentile_rank(series, lookback=100):
    """Calculate percentile rank of current value vs lookback period."""
    n = len(series)
    pr = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = series[i-lookback+1:i+1]
        current = series[i]
        pr[i] = np.sum(window < current) / lookback * 100.0
    
    return pr

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
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct_rank = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track RSI cross for entry timing
    prev_rsi = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === ADX FILTER - Only trade when there's some trend ===
        trend_present = adx[i] > 20.0
        
        # === BOLLINGER BAND SQUEEZE ===
        # BB width in bottom 20th percentile = squeeze (potential breakout)
        bb_squeeze = bb_width_pct_rank[i] < 20.0 if not np.isnan(bb_width_pct_rank[i]) else False
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI was < 45, now crosses above 50 (pullback complete in uptrend)
        # Short: RSI was > 55, now crosses below 50 (rally complete in downtrend)
        rsi_cross_long = prev_rsi < 45.0 and rsi[i] > 50.0
        rsi_cross_short = prev_rsi > 55.0 and rsi[i] < 50.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Weekly bull trend + ADX confirms + RSI pullback complete
        if bull_trend_1w and trend_present and rsi_cross_long:
            new_signal = SIZE
        
        # SHORT ENTRY: Weekly bear trend + ADX confirms + RSI rally complete
        elif bear_trend_1w and trend_present and rsi_cross_short:
            new_signal = -SIZE
        
        # BB SQUEEZE BREAKOUT ENHANCEMENT
        # If squeeze detected, allow entries with lower ADX threshold
        if bb_squeeze and not trend_present:
            # Relaxed ADX for squeeze breakout
            if bull_trend_1w and rsi_cross_long:
                new_signal = SIZE
            elif bear_trend_1w and rsi_cross_short:
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
        # Exit if weekly trend reverses against position
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
        prev_rsi = rsi[i]
    
    return signals