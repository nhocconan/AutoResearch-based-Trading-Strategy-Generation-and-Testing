#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume and ATR regime filter
# - Primary: 6h timeframe for moderate trade frequency (target: 75-200 trades over 4 years)
# - HTF: 12h for Camarilla pivot levels (R3/S3, R4/S4) and volume confirmation
# - Long: Price breaks above R4 with volume > 1.5x 20-period MA and ATR > 40th percentile
# - Short: Price breaks below S4 with volume > 1.5x 20-period MA and ATR > 40th percentile
# - Exit: Price retouches R3/S3 (mean reversion) or ATR < 20th percentile (low vol)
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Camarilla levels act as support/resistance; volume/ATR filters avoid false breakouts

name = "6h_12h_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels
    # Based on previous 12h bar's high, low, close
    # HLC = (high + low + close) / 3
    # Range = high - low
    # R4 = HLC + Range * 1.1/2
    # R3 = HLC + Range * 1.1/4
    # S3 = HLC - Range * 1.1/4
    # S4 = HLC - Range * 1.1/2
    hlc = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    r4 = hlc + range_12h * 1.1 / 2
    r3 = hlc + range_12h * 1.1 / 4
    s3 = hlc - range_12h * 1.1 / 4
    s4 = hlc - range_12h * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4)
    r3_6h = align_htf_to_ltf(prices, df_12h, r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR percentile rank (using 30-bar lookback)
    atr_percentile = pd.Series(atr_12h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_6h = align_htf_to_ltf(prices, df_12h, atr_percentile)
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_6h = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Align current 12h volume for volume spike detection
    volume_12h_6h = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(atr_percentile_6h[i]) or 
            np.isnan(volume_ma_20_12h_6h[i]) or
            np.isnan(volume_12h_6h[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # 12h volatility regime: ATR > 40th percentile (avoid low-vol chop)
        vol_regime = atr_percentile_6h[i] > 40
        
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h_6h[i] > 1.5 * volume_ma_20_12h_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R4 + volume spike + vol regime
            if (close_6h[i] > r4_6h[i] and volume_spike and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S4 + volume spike + vol regime
            elif (close_6h[i] < s4_6h[i] and volume_spike and vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retouches R3/S3 (mean reversion at stronger levels)
            # 2. ATR falls below 20th percentile (low volatility regime)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < r3_6h[i] or  # Price retraced below R3
                    atr_percentile_6h[i] < 20  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > s3_6h[i] or  # Price retraced above S3
                    atr_percentile_6h[i] < 20  # Low volatility regime
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals