#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Camarilla pivot levels with 12h volume confirmation
    # Fade at R3/S3 levels in ranging markets, breakout continuation at R4/S4 in trending markets
    # Uses 12h ADX > 25 to distinguish trending vs ranging regimes
    # Target: 12-25 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for volume confirmation and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h ADX (14-period) for regime detection
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        plus_dm_smoothed = np.zeros_like(high)
        minus_dm_smoothed = np.zeros_like(high)
        plus_dm_smoothed[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smoothed[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Get 12h volume for confirmation (20-period average)
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    camarilla_pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(high_1d)):
        # Typical price for pivot calculation
        typical_price = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        camarilla_pivot[i] = typical_price
        range_ = high_1d[i-1] - low_1d[i-1]
        
        # Camarilla levels
        camarilla_r4[i] = camarilla_pivot[i] + range_ * 1.5
        camarilla_r3[i] = camarilla_pivot[i] + range_ * 1.25
        camarilla_s3[i] = camarilla_pivot[i] - range_ * 1.25
        camarilla_s4[i] = camarilla_pivot[i] - range_ * 1.5
    
    # Align all HTF indicators to 6h primary timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        idx_12h = i // 2  # 12h bars in 6h timeframe (2 bars per 12h)
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Regime detection: 12h ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25
        ranging = adx_12h_aligned[i] <= 25
        
        # Entry conditions based on regime
        if ranging:
            # In ranging markets: fade at R3/S3 levels
            enter_long = (close[i] <= camarilla_s3_aligned[i]) and volume_confirmed
            enter_short = (close[i] >= camarilla_r3_aligned[i]) and volume_confirmed
            # Exit at pivot or opposite S3/R3
            exit_long = position == 1 and close[i] >= camarilla_pivot_aligned[i]
            exit_short = position == -1 and close[i] <= camarilla_pivot_aligned[i]
        else:
            # In trending markets: breakout continuation at R4/S4 levels
            enter_long = (close[i] >= camarilla_r4_aligned[i]) and volume_confirmed
            enter_short = (close[i] <= camarilla_s4_aligned[i]) and volume_confirmed
            # Exit at opposite R4/S4 or when trend weakens
            exit_long = position == 1 and (close[i] <= camarilla_s4_aligned[i] or adx_12h_aligned[i] < 20)
            exit_short = position == -1 and (close[i] >= camarilla_r4_aligned[i] or adx_12h_aligned[i] < 20)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_12h_1d_camarilla_pivot_adx_volume_v1"
timeframe = "6h"
leverage = 1.0