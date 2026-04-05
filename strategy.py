#!/usr/bin/env python3
"""
Experiment #8274: 1-hour mean reversion with 4h/1d trend filter.
Hypothesis: In choppy markets (ADX < 25), price reverts to the 4h VWAP. 
Use 1d trend (price above/below 200 EMA) for directional bias, and 4h ADX 
to filter ranging vs trending conditions. Enter on 1h when price deviates 
>1.5σ from 4h VWAP with volume confirmation. Target 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8274_1h_vwap_reversion_4h_adx_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
VWAP_PERIOD = 20          # 4h VWAP period
ADX_PERIOD = 14           # 4h ADX for regime detection
ADX_THRESHOLD = 25        # Below 25 = ranging (good for mean reversion)
EMA_TREND_PERIOD = 200    # 1d EMA for trend filter
DEV_THRESHOLD = 1.5       # Standard deviations for entry
VOLUME_MA_PERIOD = 20     # Volume confirmation
VOLUME_THRESHOLD = 1.2    # Volume must be above average
SIGNAL_SIZE = 0.20        # Position size
ATR_PERIOD = 14           # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5 # Stop loss distance

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h VWAP (typical price * volume cumulative)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    vp_4h = typical_price_4h * df_4h['volume']
    cum_vp_4h = vp_4h.rolling(window=VWAP_PERIOD, min_periods=VWAP_PERIOD).sum()
    cum_vol_4h = df_4h['volume'].rolling(window=VWAP_PERIOD, min_periods=VWAP_PERIOD).sum()
    vwap_4h = cum_vp_4h / cum_vol_4h
    vwap_4h_vals = vwap_4h.values
    
    # Calculate 4h ADX for regime detection
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_4h = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_4h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_4h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA200: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_200, 1, -1)  # 1=bullish, -1=bearish
    
    # Align HTF indicators to 1h timeframe
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_vals)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h ATR for stop loss
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(np.maximum(tr1_h, tr2_h), tr3_h)
    atr_h = pd.Series(tr_h).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_PERIOD, ADX_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(vwap_4h_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime from 4h ADX
        ranging = adx_aligned[i] < ADX_THRESHOLD  # ADX < 25 = ranging
        
        # Determine trend bias from 1d EMA200
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA200
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Deviation from 4h VWAP (calculate rolling std)
        if i >= VWAP_PERIOD:
            # Use last VWAP_PERIOD of 1h prices to calculate deviation from 4h VWAP
            price_slice = close[i-VWAP_PERIOD+1:i+1]
            vwap_slice = vwap_4h_aligned[i-VWAP_PERIOD+1:i+1]
            # Only use valid VWAP values
            valid_mask = ~np.isnan(vwap_slice)
            if np.sum(valid_mask) >= VWAP_PERIOD//2:  # At least half valid
                dev = price_slice - vwap_slice
                dev_valid = dev[valid_mask]
                if len(dev_valid) > 0:
                    std_dev = np.std(dev_valid)
                    if std_dev > 0:
                        current_dev = close[i] - vwap_4h_aligned[i]
                        z_score = current_dev / std_dev
                    else:
                        z_score = 0
                else:
                    z_score = 0
            else:
                z_score = 0
        else:
            z_score = 0
        
        # Entry conditions: only in ranging markets with volume confirmation
        long_entry = ranging and bull_bias and volume_confirmed and (z_score < -DEV_THRESHOLD)
        short_entry = ranging and bear_bias and volume_confirmed and (z_score > DEV_THRESHOLD)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_h[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_h[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals