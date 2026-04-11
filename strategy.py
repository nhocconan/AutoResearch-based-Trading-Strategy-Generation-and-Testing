#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout/continuation + 12h volume confirmation + ATR volatility filter
# - Camarilla levels (R3,R4,S3,S4) from 12h HTF act as key support/resistance
# - Long when price breaks above R4 with volume confirmation (continuation)
# - Long when price rebounds from S3 with volume confirmation (mean reversion)
# - Short when price breaks below S4 with volume confirmation (continuation)
# - Short when price rebounds from R3 with volume confirmation (mean reversion)
# - ATR filter: only trade when ATR(14) > 0.3 * ATR(50) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts at R4/S4) and bear (mean reversion at R3/S3) markets
# - 12h HTF provides reliable Camarilla calculation and volume confirmation

name = "6h_12h_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for Camarilla pivots and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h Camarilla pivots (based on previous day's high, low, close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    camarilla_r4 = close_12h + (high_12h - low_12h) * 1.1 / 2
    camarilla_r3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_s3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    camarilla_s4 = close_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Pre-compute 12h volume SMA (20-period)
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 12h ATR for volatility filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_12h = pd.Series(tr_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average (using 12h aligned volume)
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # ATR filter: trade only when short-term ATR > 0.3 * long-term ATR (avoid low volatility)
        atr_filter = atr_14_aligned[i] > 0.3 * atr_50_aligned[i]
        
        # Camarilla levels from previous 12h bar (to avoid look-ahead)
        r4 = camarilla_r4_aligned[i-1]
        r3 = camarilla_r3_aligned[i-1]
        s3 = camarilla_s3_aligned[i-1]
        s4 = camarilla_s4_aligned[i-1]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume confirmation
        if price_close > r4 and vol_confirm and atr_filter:
            enter_long = True
        
        # Long mean reversion: price rebounds from S3 with volume confirmation
        if price_close > s3 and price_low <= s3 and vol_confirm and atr_filter:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume confirmation
        if price_close < s4 and vol_confirm and atr_filter:
            enter_short = True
        
        # Short mean reversion: price rebounds from R3 with volume confirmation
        if price_close < r3 and price_high >= r3 and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below S3 OR volatility collapses
            exit_long = (price_close < s3) or (not atr_filter)
        elif position == -1:
            # Exit short if price breaks above R3 OR volatility collapses
            exit_short = (price_close > r3) or (not atr_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals