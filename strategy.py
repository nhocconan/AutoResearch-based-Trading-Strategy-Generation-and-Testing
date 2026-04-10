#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w EMA(50) trend filter and volume confirmation
# - Uses weekly pivot levels (R3/S3, R4/S4) calculated from prior week's OHLC
# - Long: Price breaks above weekly R3 + weekly EMA(50) rising + 6h volume > 1.5x 20-period average
# - Short: Price breaks below weekly S3 + weekly EMA(50) falling + same volume confirmation
# - Exit: Close-based reversal - exit when price crosses weekly pivot point (PP) in opposite direction
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 6h
# - Position sizing: 0.25 (discrete level)
# - Weekly EMA(50) provides strong trend filter to avoid counter-trend trades in both bull/bear
# - Camarilla levels from weekly timeframe provide institutional support/resistance
# - Volume confirmation ensures breakout validity
# - Target: 60-120 total trades over 4 years (15-30/year) to stay within limits
# - Works in bull markets via breakouts, in bear via shorting failed rallies at resistance

name = "6h_1w_camarilla_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla pivot levels from prior week's OHLC
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    # R4 = PP + (H - L) * 1.1
    # S4 = PP - (H - L) * 1.1
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    pp_1w = typical_price_1w
    r3_1w = pp_1w + range_1w * 1.1 / 2.0
    s3_1w = pp_1w - range_1w * 1.1 / 2.0
    r4_1w = pp_1w + range_1w * 1.1
    s4_1w = pp_1w - range_1w * 1.1
    
    # Align weekly levels to 6h (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly EMA slope (rising/falling)
    ema_slope = np.diff(ema_50_aligned, prepend=np.nan)
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate 6h volume moving average (20-period)
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_6h)
    
    # Calculate 6h ATR (14-period) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is 0 (no previous close)
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_6h = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h data
        close_price = close_6h[i]
        volume_current = volume_6h[i]
        volume_confirmation = volume_current > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above weekly R3 + EMA rising + volume confirmation
            if (close_price > r3_aligned[i] and ema_rising[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below weekly S3 + EMA falling + volume confirmation
            elif (close_price < s3_aligned[i] and ema_falling[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.5 * atr_14_6h[i]
                # Exit conditions: price < weekly pivot point OR stoploss hit
                if close_price < pp_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_14_6h[i]
                # Exit conditions: price > weekly pivot point OR stoploss hit
                if close_price > pp_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals