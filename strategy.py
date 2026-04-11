#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot + 12h Volume Spike + Regime Filter
# - Camarilla levels from 12h: R3/S3 for mean reversion, R4/S4 for breakout
# - Volume spike confirmation: 6h volume > 2.0x 20-period average
# - Regime filter: 12h ADX > 25 for trending (breakout), ADX < 20 for ranging (mean reversion)
# - Long conditions: 
#   * Trending (ADX>25): close > R4 AND volume spike
#   * Ranging (ADX<20): close < S3 AND volume spike AND RSI(14) < 30
# - Short conditions:
#   * Trending (ADX>25): close < S4 AND volume spike
#   * Ranging (ADX<20): close > R3 AND volume spike AND RSI(14) > 70
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla provides mathematically derived support/resistance levels
# - Volume confirmation ensures institutional participation
# - Regime filter adapts strategy to market conditions (trending vs ranging)

name = "6h_12h_camarilla_volume_regime_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Pre-compute 12h Camarilla levels (based on previous day's range)
    # Camarilla uses previous period's high, low, close
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed - based on completed 12h bar)
    r3_12h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_12h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_12h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute 12h ADX for regime filter
    # Calculate True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = np.where((df_12h['high'] - df_12h['high'].shift(1)) > (df_12h['low'].shift(1) - df_12h['low']),
                       np.maximum(df_12h['high'] - df_12h['high'].shift(1), 0), 0)
    dm_minus = np.where((df_12h['low'].shift(1) - df_12h['low']) > (df_12h['high'] - df_12h['high'].shift(1)),
                        np.maximum(df_12h['low'].shift(1) - df_12h['low'], 0), 0)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI and DX
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 6h RSI for ranging regime signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Regime determination
        adx_current = adx_aligned[i]
        is_trending = adx_current > 25
        is_ranging = adx_current < 20
        
        # Camarilla levels
        r3_level = r3_12h[i]
        s3_level = s3_12h[i]
        r4_level = r4_12h[i]
        s4_level = s4_12h[i]
        
        # RSI for ranging regime
        rsi_current = rsi_values[i]
        
        # Initialize signal flags
        enter_long = False
        enter_short = False
        exit_long = False
        exit_short = False
        
        # Trading logic based on regime
        if is_trending:
            # Trending regime: breakout strategy
            # Long: price breaks above R4 with volume
            if close_current > r4_level and vol_confirm:
                enter_long = True
            # Short: price breaks below S4 with volume
            elif close_current < s4_level and vol_confirm:
                enter_short = True
            
            # Exit: reverse breakout or loss of volume
            if position == 1:
                exit_long = close_current < r3_level or not vol_confirm
            elif position == -1:
                exit_short = close_current > s3_level or not vol_confirm
                
        elif is_ranging:
            # Ranging regime: mean reversion strategy
            # Long: price at S3 with oversold RSI and volume
            if close_current < s3_level and rsi_current < 30 and vol_confirm:
                enter_long = True
            # Short: price at R3 with overbought RSI and volume
            elif close_current > r3_level and rsi_current > 70 and vol_confirm:
                enter_short = True
            
            # Exit: reverse to opposite level or RSI normalization
            if position == 1:
                exit_long = close_current > r3_level or rsi_current > 50
            elif position == -1:
                exit_short = close_current < s3_level or rsi_current < 50
        
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