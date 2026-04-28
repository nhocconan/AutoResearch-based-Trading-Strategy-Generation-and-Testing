#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 with 1d EMA34 uptrend and volume > 2.0x 24-bar average.
# Enter short when price breaks below Camarilla S3 with 1d EMA34 downtrend and volume > 2.0x 24-bar average.
# Exit when price retraces to the 12h EMA21.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 ensures higher timeframe alignment;
# volume confirmation filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Use previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + 1.1 * camarilla_range / 2
    camarilla_S3 = prev_close - 1.1 * camarilla_range / 2
    
    # Volume confirmation: >2.0x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    # Exit condition: 12h EMA21
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for volume MA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, EMA34 up, volume confirm
            if price > camarilla_R3[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S3, EMA34 down, volume confirm
            elif price < camarilla_S3[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at EMA21
            if price <= ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at EMA21
            if price >= ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals