#!/usr/bin/env python3
"""
Experiment #560: 30m Bollinger Squeeze Breakout with 4h HMA Trend Bias

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. 30m timeframe captures intraday momentum without excessive noise
2. Bollinger Band squeeze (low BW) precedes major breakouts - proven edge
3. 4h HMA trend bias prevents counter-trend entries (major failure mode in #554)
4. Volume confirmation filters false breakouts
5. ADX>18 filter (loose) ensures we trade but avoid worst chop
6. ATR stoploss at 2.5x protects against 2022-style crashes

Why this should work on 30m:
- BB squeeze catches volatility expansion BEFORE the move
- 4h HMA is proven in successful strategies (mtf_4h_regime_chop works)
- Volume spike confirms genuine breakout vs fakeout
- Loose ADX>18 (not >40) ensures we generate trades (learned from #551/#555/#557)
- 2.5*ATR stoploss with trailing protects capital

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_breakout_4h_hma_volume_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, bandwidth.values, sma.values

def calculate_volume_spike(volume, period=20):
    """Calculate volume spike ratio (current vs rolling avg)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg
    return vol_ratio.values

def calculate_bb_squeeze(bandwidth, lookback=60):
    """Detect BB squeeze - bandwidth at low percentile."""
    bw_s = pd.Series(bandwidth)
    # Calculate percentile rank of current bandwidth over lookback
    bb_percentile = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5,
        raw=False
    )
    return bb_percentile.values

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
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_spike(volume, 20)
    bb_squeeze = calculate_bb_squeeze(bb_bandwidth, 60)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_squeeze[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === BB SQUEEZE DETECTION (low volatility = impending breakout) ===
        squeeze_active = bb_squeeze[i] < 0.30  # Bandwidth in bottom 30% of 60-bar range
        
        # === BREAKOUT DETECTION ===
        breakout_long = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        breakout_short = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3  # 30% above average volume
        
        # === ADX FILTER (trend strength - loose threshold for trade generation) ===
        trend_strong = adx_14[i] > 18  # Loose filter to ensure trades (learned from #551/#555)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: BB breakout + squeeze preceded it + volume confirms + 4h bullish + ADX
        if breakout_long and bull_bias and trend_strong:
            # Either squeeze preceded OR volume spike confirms
            if squeeze_active or volume_confirmed:
                new_signal = SIZE
        
        # Short: BB breakout + squeeze preceded it + volume confirms + 4h bearish + ADX
        elif breakout_short and bear_bias and trend_strong:
            # Either squeeze preceded OR volume spike confirms
            if squeeze_active or volume_confirmed:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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