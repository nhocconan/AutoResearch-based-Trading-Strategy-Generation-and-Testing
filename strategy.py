#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Long when price breaks above R3 AND 1d EMA34 uptrend AND volume spike
# Short when price breaks below S3 AND 1d EMA34 downtrend AND volume spike
# Uses discrete position sizing (0.30) to minimize fee churn. Works in both bull/bear by following 1d trend.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v2"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (use prior completed 1d bar)
    # align_htf_to_ltf with additional_delay_bars=1 ensures we use yesterday's close
    prev_close = align_htf_to_ltf(prices, df_1d, close_1d, additional_delay_bars=1)
    prev_high = align_htf_to_ltf(prices, df_1d, df_1d['high'].values, additional_delay_bars=1)
    prev_low = align_htf_to_ltf(prices, df_1d, df_1d['low'].values, additional_delay_bars=1)
    
    # Camarilla levels: R3/S3 = close ± 1.1*(high-low)/2
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_R3 = R3[i]
        curr_S3 = S3[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > R3 AND bullish regime
                if curr_close > curr_R3 and is_bullish_regime:
                    signals[i] = 0.30
                    position = 1
                # Bearish entry: price < S3 AND bearish regime
                elif curr_close < curr_S3 and is_bearish_regime:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit when price < S3 or regime changes
            if curr_close < curr_S3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit when price > R3 or regime changes
            if curr_close > curr_R3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals