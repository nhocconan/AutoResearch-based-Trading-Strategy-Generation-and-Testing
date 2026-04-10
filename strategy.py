#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d ADX trend filter
# - Primary: 6h price breaking above/below Camarilla R4/S4 levels derived from prior 1d candle
# - Volume filter: 12h volume > 1.8x 20-period volume MA to ensure participation
# - Trend filter: 1d ADX > 20 to avoid choppy markets
# - Exit: Price reverses to Camarilla R3/S3 levels (profit target) or opposite S4/R4 (stop)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms breakout, ADX avoids false signals

name = "6h_12h_1d_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate prior 1d Camarilla levels (using previous day's OHLC)
    # Shift by 1 to use completed day only
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r4 = prev_close_1d + camarilla_range * 1.1 / 2
    camarilla_r3 = prev_close_1d + camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close_1d - camarilla_range * 1.1 / 4
    camarilla_s4 = prev_close_1d - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 12h volume confirmation: volume > 1.8x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1d ADX for trend filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    
    # Handle first element
    high_low_1d[0] = high_1d[0] - low_1d[0]
    high_close_1d[0] = np.abs(high_1d[0] - close_1d[0])
    low_close_1d[0] = np.abs(low_1d[0] - close_1d[0])
    
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_smoothed = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.8x 20-period volume MA
        volume_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = volume_12h_current[i] > 1.8 * volume_ma_20_12h_aligned[i]
        
        # Trend filter: ADX > 20 to avoid choppy markets
        trending_market = adx_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R4 + vol confirmation + trending market
            if (close[i] > camarilla_r4_aligned[i] and 
                vol_confirm and trending_market):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla S4 + vol confirmation + trending market
            elif (close[i] < camarilla_s4_aligned[i] and 
                  vol_confirm and trending_market):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # Long: price reaches R3 (profit target) or crosses below S4 (stop)
            # Short: price reaches S3 (profit target) or crosses above R4 (stop)
            if position == 1:  # Long position
                if close[i] >= camarilla_r3_aligned[i] or close[i] <= camarilla_s4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] <= camarilla_s3_aligned[i] or close[i] >= camarilla_r4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals