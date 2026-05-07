#!/usr/bin/env python3
"""
6H_RSI_TREND_FILTER_12H_CCI_CONFIRMATION
Hypothesis: Use 6h RSI with 12h CCI trend filter to capture mean-reversion in range markets and trend continuation in strong trends.
Long when 6h RSI < 30 and 12h CCI > -100 (bullish bias); Short when 6h RSI > 70 and 12h CCI < 100 (bearish bias).
Volume confirmation: current volume > 1.3x 20-period average volume.
Designed to work in both bull (trend continuation) and bear (mean reversion) markets by combining momentum oscillator with trend filter.
"""
name = "6H_RSI_TREND_FILTER_12H_CCI_CONFIRMATION"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for CCI trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h CCI (20-period)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    sma_tp = typical_price_12h.rolling(window=20, min_periods=20).mean()
    mad = typical_price_12h.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price_12h - sma_tp) / (0.015 * mad.replace(0, np.nan))
    cci = cci.fillna(0).values
    cci_aligned = align_htf_to_ltf(prices, df_12h, cci)
    
    # Calculate 6h RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Volume filter: current volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(cci_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold and 12h CCI shows bullish bias
            if rsi[i] < 30 and cci_aligned[i] > -100 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought and 12h CCI shows bearish bias
            elif rsi[i] > 70 and cci_aligned[i] < 100 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long when RSI returns to neutral or turns bearish
                if rsi[i] >= 50 or cci_aligned[i] < -100:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                # Exit short when RSI returns to neutral or turns bullish
                if rsi[i] <= 50 or cci_aligned[i] > 100:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals