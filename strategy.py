#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; EMA50 on 1w ensures alignment with major trend.
# Volume confirmation (>1.5x 20-period average) filters weak breakouts.
# ATR-based trailing stop (2.5x ATR) manages risk and adapts to volatility.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Donchian calculation (primary TF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data for EMA50 trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Donchian Channels (20-period) on 1d ===
    period = 20
    donchian_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    
    # === EMA50 on 1w for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation: 1.5x 20-period average volume on 1d ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === ATR for trailing stop (20-period) on 1d ===
    atr_period = 20
    tr1 = pd.Series(high_1d).subtract(pd.Series(low_1d)).abs()
    tr2 = pd.Series(high_1d).subtract(pd.Series(close_1d).shift(1)).abs()
    tr3 = pd.Series(low_1d).subtract(pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # === Align all indicators to 1d timeframe ===
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=0)  # EMA confirmed on close
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        atr_val = atr_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > vol_ma * 1.5
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Trail stop: exit if price drops 2.5*ATR from extreme
            if atr_val > 0 and price < extreme_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update extreme price (lowest since entry)
            if price < extreme_price or extreme_price == 0:
                extreme_price = price
            # Trail stop: exit if price rises 2.5*ATR from extreme
            if atr_val > 0 and price > extreme_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND above 1w EMA50 AND volume confirmation
            if price > upper and price > ema50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: price breaks below Donchian low AND below 1w EMA50 AND volume confirmation
            elif price < lower and price < ema50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                extreme_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_ATRTrail"
timeframe = "1d"
leverage = 1.0