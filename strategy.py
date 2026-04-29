#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 (strong resistance turned support) with volume spike AND 12h EMA50 uptrend
# Short when price breaks below S3 (strong support turned resistance) with volume spike AND 12h EMA50 downtrend
# Uses proven Camarilla pivot structure from DB winners with tight entry conditions to avoid overtrading
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing strong breakouts

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from daily data (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    high_1d = df_12h['high'].values  # Using 12h high/low/close as proxy for daily (more frequent updates)
    low_1d = df_12h['low'].values
    close_1d = df_12h['close'].values
    
    # Camarilla levels: based on previous period's range
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # We calculate these on 12h data and align to 4h
    range_12h = high_1d - low_1d
    camarilla_r3 = close_1d + (range_12h * 1.1 / 2)
    camarilla_s3 = close_1d - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 12h EMA50, bearish if price < 12h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 with volume AND bullish regime
                if curr_high > curr_r3 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 with volume AND bearish regime
                elif curr_low < curr_s3 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below S3 or regime changes
            if curr_low < curr_s3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price rises above R3 or regime changes
            if curr_high > curr_r3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals