#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_VolumeSpike_v3
Hypothesis: Daily Camarilla R1/S1 breakouts filtered by weekly EMA50 trend regime and 4h volume spikes. 
Weekly EMA50 provides robust trend filter to avoid whipsaws in sideways markets. Volume confirmation ensures 
institutional participation. Discrete sizing (0.25) targets 15-25 trades/year. Works in bull/bear via regime 
adaptation: follows trend in strong regimes, avoids false breakouts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, 1w for trend regime)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) based on PREVIOUS 1d bar's OHLC ===
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d[0] = prev_low_1d[0] = prev_close_1d[0] = np.nan  # first bar invalid
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = pivot_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    s1_1d = pivot_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 15m timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w EMA50 for trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1w EMA50 slope for regime classification ===
    ema_slope_1w = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    # Regime: trending up if slope > 0.05% of price, trending down if < -0.05%, else chop
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    slope_threshold = 0.0005 * close_1w_aligned
    trending_up = ema_slope_1w_aligned > slope_threshold
    trending_down = ema_slope_1w_aligned < -slope_threshold
    ranging = ~(trending_up | trending_down)
    
    # === 4h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # === 4h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_slope_1w_aligned[i]) or 
            np.isnan(volume_confirmed[i]) or np.isnan(atr[i]) or 
            np.isnan(trending_up[i]) or np.isnan(trending_down[i]) or np.isnan(ranging[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        
        # Regime flags
        is_trending_up = trending_up[i]
        is_trending_down = trending_down[i]
        is_ranging = ranging[i]
        
        if position == 0:
            # Regime-adaptive entry conditions
            if is_trending_up:
                # Bull regime: favor longs, require alignment with uptrend
                long_condition = (price > r1_val) and vol_conf and (price > ema_50_1w_aligned[i])
                short_condition = (price < s1_val) and vol_conf and (price < ema_50_1w_aligned[i] * 0.995)  # stricter for shorts
            elif is_trending_down:
                # Bear regime: favor shorts, require alignment with downtrend
                long_condition = (price > r1_val) and vol_conf and (price > ema_50_1w_aligned[i] * 1.005)  # stricter for longs
                short_condition = (price < s1_val) and vol_conf and (price < ema_50_1w_aligned[i])
            else:  # ranging regime
                # Chop regime: trade both directions but require stronger volume confirmation
                long_condition = (price > r1_val) and vol_conf and (price > ema_50_1w_aligned[i] * 1.002)
                short_condition = (price < s1_val) and vol_conf and (price < ema_50_1w_aligned[i] * 0.998)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout) or strong adverse move
                elif price < s1_val or price < ema_50_1w_aligned[i] * 0.99:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown) or strong adverse move
                elif price > r1_val or price > ema_50_1w_aligned[i] * 1.01:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_VolumeSpike_v3"
timeframe = "1d"
leverage = 1.0