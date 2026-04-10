#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Entry: Long when price breaks above Camarilla H3 level (1d) + 1d volume > 1.3x 20-period average + ADX(14, 12h) > 20
#          Short when price breaks below Camarilla L3 level (1d) + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < Camarilla L3 level (1d), exit short when price > Camarilla H3 level (1d)
# - Stoploss: ATR-based - exit when price moves against position by 1.5 * ATR(14) on 12h
# - Position sizing: 0.25 (discrete level)
# - Camarilla pivots provide intraday support/resistance that work in ranging markets
# - Volume confirmation ensures breakout validity, ADX filter avoids choppy markets
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within HARD MAX: 200 total

name = "12h_1d_camarilla_breakout_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1d data for Camarilla and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1 / 4)
    # L3 = pivot - (range * 1.1 / 4)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    price_range = high_1d - low_1d
    camarilla_h3 = typical_price + (price_range * 1.1 / 4.0)
    camarilla_l3 = typical_price - (price_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h (using previous completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h ADX (14-period) for trend filter
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM) / EWMA(TR)
    # -DI = 100 * EWMA(-DM) / EWMA(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # Calculate directional movement
    high_diff = high_12h - np.roll(high_12h, 1)
    low_diff = np.roll(low_12h, 1) - low_12h
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Set first values to 0
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    
    # Calculate smoothed values using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    # Smooth TR, +DM, -DM
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    
    # Calculate DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    dx_12h = np.where(np.isnan(dx_12h), 0, dx_12h)  # Handle division by zero
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Calculate 12h ATR (14-period) for stoploss
    atr_14_12h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 1.3 * volume_ma_aligned[i]
        
        # Trend filter: ADX > 20 indicates trending market (avoid choppy conditions)
        trend_filter = adx_12h[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + trend filter
            if (close_price > camarilla_h3_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + trend filter
            elif (close_price < camarilla_l3_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 1.5 * atr_14_12h[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 1.5 * atr_14_12h[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals