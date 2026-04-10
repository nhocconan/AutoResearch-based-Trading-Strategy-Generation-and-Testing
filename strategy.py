#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 12h trend filter
# - Entry: Long when price breaks above Camarilla H3 (1d) + 1d volume > 1.8x 20-period average + 12h EMA(21) > EMA(50)
#          Short when price breaks below Camarilla L3 (1d) + same volume and trend filters
# - Exit: Close-based reversal - exit long when price < Camarilla L3, exit short when price > Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Volume confirmation (1.8x) balances sensitivity and false breakout reduction
# - EMA crossover provides adaptive trend filter that works in trending and ranging markets
# - Target: 100-180 total trades over 4 years (25-45/year) to stay within HARD MAX: 400 total

name = "4h_1d_12h_camarilla_breakout_volume_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 60:  # Need enough for EMA50
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data for Camarilla and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 12h data for EMA
    close_12h = df_12h['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (high + low + close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = high - low
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pp_1d + range_1d * 1.1 / 4.0  # H3 = PP + 1.1*(H-L)/4
    l3_1d = pp_1d - range_1d * 1.1 / 4.0  # L3 = PP - 1.1*(H-L)/4
    
    # Align Camarilla levels to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 12h EMA(21) and EMA(50) for trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h ATR (14-period) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = 0
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_4h = wilders_smoothing(tr_4h, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(200, n):  # Start after warmup period for EMA50
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmation = volume_1d_current > 1.8 * volume_ma_aligned[i]
        
        # Trend filter: EMA21 > EMA50 indicates uptrend, EMA21 < EMA50 indicates downtrend
        ema_trend_up = ema_21_aligned[i] > ema_50_aligned[i]
        ema_trend_down = ema_21_aligned[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + uptrend
            if (close_price > h3_1d_aligned[i] and 
                volume_confirmation and 
                ema_trend_up):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + downtrend
            elif (close_price < l3_1d_aligned[i] and 
                  volume_confirmation and 
                  ema_trend_down):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < l3_1d_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > h3_1d_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals