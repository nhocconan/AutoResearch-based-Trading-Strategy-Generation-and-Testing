#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout/fade + 12h volume confirmation + 1d ADX regime filter
# - Primary signal: Price breaks above/below Camarilla R4/S4 levels (strong breakout)
# - Fade signal: Price rejects at R3/S3 levels with volume exhaustion (mean reversion in range)
# - Volume confirmation: 12h volume > 1.3x 24-period average volume (avoid fakeouts)
# - Regime filter: 1d ADX > 25 for breakout continuation, ADX < 20 for mean reversion
# - Works in bull/bear: Breakouts work in trending markets (ADX>25), mean reversion works in ranging markets (ADX<20)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_12h_1d_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation
    volume_12h = df_12h['volume'].values
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume_12h > (1.3 * avg_volume_24)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / np.where(tr_14 != 0, tr_14, 1e-10)
    minus_di = 100 * minus_dm_14 / np.where(tr_14 != 0, tr_14, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Camarilla levels (based on previous 6h bar)
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Shift by 1 to use previous bar's OHLC (no look-ahead)
    prev_close = np.roll(close_6h, 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close[0] = close_6h[0]  # first bar
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    
    # Camarilla levels
    range_prev = prev_high - prev_low
    camarilla_h4 = prev_close + range_prev * 1.1 / 2
    camarilla_l4 = prev_close - range_prev * 1.1 / 2
    camarilla_h3 = prev_close + range_prev * 1.1 / 4
    camarilla_l3 = prev_close - range_prev * 1.1 / 4
    camarilla_h2 = prev_close + range_prev * 1.1 / 6
    camarilla_l2 = prev_close - range_prev * 1.1 / 6
    camarilla_h1 = prev_close + range_prev * 1.1 / 12
    camarilla_l1 = prev_close - range_prev * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_confirm_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            
            # Stoploss: 2.5x ATR based on recent volatility
            atr_est = np.abs(high_6h[i] - low_6h[i])  # simple ATR proxy
            if close_6h[i] < entry_price - 2.5 * atr_est:
                exit_long = True
            
            # Take profit at Camarilla H3/H4
            elif close_6h[i] >= camarilla_h3[i]:
                exit_long = True
            
            # Reverse signal: strong rejection at H3 with volume
            elif (close_6h[i] <= camarilla_h3[i] * 1.002 and  # touched H3
                  volume_confirm_aligned[i] and 
                  adx_aligned[i] < 25):  # weak trend - mean reversion
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            
            # Stoploss: 2.5x ATR based on recent volatility
            atr_est = np.abs(high_6h[i] - low_6h[i])  # simple ATR proxy
            if close_6h[i] > entry_price + 2.5 * atr_est:
                exit_short = True
            
            # Take profit at Camarilla L3/L4
            elif close_6h[i] <= camarilla_l3[i]:
                exit_short = True
            
            # Reverse signal: strong rejection at L3 with volume
            elif (close_6h[i] >= camarilla_l3[i] * 0.998 and  # touched L3
                  volume_confirm_aligned[i] and 
                  adx_aligned[i] < 25):  # weak trend - mean reversion
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries
            current_price = close_6h[i]
            
            # Breakout continuation: price breaks R4/S4 with volume and strong trend (ADX>25)
            if volume_confirm_aligned[i] and adx_aligned[i] > 25:
                if current_price > camarilla_h4[i]:  # break above R4
                    position = 1
                    entry_price = current_price
                    signals[i] = 0.25
                elif current_price < camarilla_l4[i]:  # break below S4
                    position = -1
                    entry_price = current_price
                    signals[i] = -0.25
            
            # Mean reversion: price rejects at R3/S3 with volume in ranging market (ADX<20)
            elif adx_aligned[i] < 20:
                if volume_confirm_aligned[i]:
                    # Long: rejection at S3
                    if (current_price >= camarilla_l3[i] * 0.998 and  # touched L3
                        current_price <= camarilla_l3[i] * 1.002):
                        position = 1
                        entry_price = current_price
                        signals[i] = 0.25
                    # Short: rejection at R3
                    elif (current_price >= camarilla_h3[i] * 0.998 and  # touched H3
                          current_price <= camarilla_h3[i] * 1.002):
                        position = -1
                        entry_price = current_price
                        signals[i] = -0.25
    
    return signals