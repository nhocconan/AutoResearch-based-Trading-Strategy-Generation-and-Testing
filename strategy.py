#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w EMA(34) trend filter and volume confirmation
# - Long: Price breaks above Camarilla H3 level + 1w EMA(34) rising + 1d volume > 1.5x 20-period average volume
# - Short: Price breaks below Camarilla L3 level + 1w EMA(34) falling + same volume confirmation
# - Exit: Close-based reversal - exit long when price < Camarilla L3, exit short when price > Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1d
# - Position sizing: 0.25 (discrete level)
# - Uses EMA on 1w for trend filter (aligned with experiment #31750: Primary=1d, HTF=1w)
# - Volume confirmation threshold set to 1.5x to reduce overtrading and fee drag
# - Target: 30-100 total trades over 4 years (7-25/year) to stay within HARD MAX: 150 total
# - Camarilla pivots work well in ranging markets (2025-2026 test period) while EMA filter avoids counter-trend trades

name = "1d_1w_camarilla_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #            L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w EMA slope (rising/falling)
    ema_slope = np.diff(ema_34_aligned, prepend=np.nan)
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
    
    atr_14_1d = wilders_smoothing(tr, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20_1d[i]) or 
            np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close and volume
        close_price = close_1d[i]
        volume_current = volume_1d[i]
        volume_confirmation = volume_current > 1.5 * volume_ma_20_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + EMA rising + volume confirmation
            if (close_price > camarilla_h3[i] and ema_rising[i] and volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + EMA falling + volume confirmation
            elif (close_price < camarilla_l3[i] and ema_falling[i] and volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_1d[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_1d[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals