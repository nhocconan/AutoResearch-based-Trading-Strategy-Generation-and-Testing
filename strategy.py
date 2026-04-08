#!/usr/bin/env python3
# [24959] 6h_1d_camarilla_pivot_v2
# Hypothesis: 6-hour Camarilla pivot reversal strategy. Uses 1-day Camarilla levels to identify
# key support/resistance zones. Takes mean-reversion trades at R3/S3 levels with confirmation
# from 6-hour RSI extremes. In trending markets (ADX > 25), takes breakout trades at R4/S4.
# Works in both bull and bear markets by adapting to regime: mean reversion in range,
# breakout in trend. Targets 15-35 trades/year per symbol with disciplined risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize Camarilla arrays
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_h5[i] = prev_close + range_val * 1.1 / 2
            camarilla_h4[i] = prev_close + range_val * 1.1
            camarilla_h3[i] = prev_close + range_val * 1.1 / 4
            camarilla_l3[i] = prev_close - range_val * 1.1 / 4
            camarilla_l4[i] = prev_close - range_val * 1.1
            camarilla_l5[i] = prev_close - range_val * 1.1 / 2
    
    # Calculate 6-hour RSI (14-period) for momentum confirmation
    def rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(arr, np.nan)
        avg_loss = np.full_like(arr, np.nan)
        
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_6h = rsi(close, 14)
    
    # Calculate ADX (14-period) for regime detection
    def adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed averages
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
            dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
            
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(atr, np.nan)
        minus_di = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        for i in range(period-1, len(atr)):
            if atr[i] != 0:
                plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
                minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX
        adx_val = np.full_like(dx, np.nan)
        for i in range(2*period-2, len(dx)):
            if not np.isnan(dx[i-period+1:i+1]).any():
                adx_val[i] = np.mean(dx[i-period+1:i+1])
        
        return adx_val
    
    adx_6h = adx(high, low, close, 14)
    
    # Align Camarilla levels to 6-hour timeframe
    camarilla_h5_6h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_6h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for RSI/ADX
        # Skip if data not ready
        if (np.isnan(rsi_6h[i]) or np.isnan(adx_6h[i]) or
            np.isnan(camarilla_h3_6h[i]) or np.isnan(camarilla_l3_6h[i]) or
            np.isnan(camarilla_h4_6h[i]) or np.isnan(camarilla_l4_6h[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Mean reversion: price reaches opposite Camarilla level (H3 for longs)
            if price >= camarilla_h3_6h[i]:
                exit_long = True
            # Stop loss: price breaks below L4 with strong momentum
            elif price < camarilla_l4_6h[i] and rsi_6h[i] < 30:
                exit_long = True
            # Trend following exit: ADX weak and price breaks H4
            elif adx_6h[i] < 20 and price > camarilla_h4_6h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Mean reversion: price reaches opposite Camarilla level (L3 for shorts)
            if price <= camarilla_l3_6h[i]:
                exit_short = True
            # Stop loss: price breaks above H4 with strong momentum
            elif price > camarilla_h4_6h[i] and rsi_6h[i] > 70:
                exit_short = True
            # Trend following exit: ADX weak and price breaks L4
            elif adx_6h[i] < 20 and price < camarilla_l4_6h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat - look for new entries
            # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
            is_trending = adx_6h[i] > 25
            is_ranging = adx_6h[i] < 20
            
            if is_ranging:
                # Mean reversion entries at H3/L3 with RSI extremes
                # Long: price at L3 with oversold RSI
                if price <= camarilla_l3_6h[i] and rsi_6h[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short: price at H3 with overbought RSI
                elif price >= camarilla_h3_6h[i] and rsi_6h[i] > 70:
                    position = -1
                    signals[i] = -0.25
            elif is_trending:
                # Breakout entries at H4/L4 with momentum confirmation
                # Long: break above H4 with bullish momentum
                if price > camarilla_h4_6h[i] and rsi_6h[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: break below L4 with bearish momentum
                elif price < camarilla_l4_6h[i] and rsi_6h[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals