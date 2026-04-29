#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + RSI(2) Mean Reversion
# BBW < 20th percentile = low volatility squeeze → mean reversion regime
# RSI(2) < 10 = extreme oversold → long entry
# RSI(2) > 90 = extreme overbought → short entry
# Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
# Volume confirmation: volume > 1.3x 20-period average
# Designed for low-frequency, high-conviction trades (~20-40/year) to minimize fee drag
# Works in both bull/bear markets by combining volatility regime with mean reversion

name = "6h_BBW_RSI2_MeanRev_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) for BBW regime
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # Percentage width
    
    # Calculate BBW percentile rank (50-period lookback)
    bbw_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate RSI(2) for mean reversion signals
    rsi_period = 2
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, rsi_period, 20) + 5  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bbw_rank[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bbw_rank = bbw_rank[i]
        curr_rsi = rsi_values[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: mean reversion complete or regime change
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (> 50) or BBW regime ends (breakout)
            if curr_rsi > 50 or curr_bbw_rank > 80:  # BBW expansion = breakout potential
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (< 50) or BBW regime ends
            if curr_rsi < 50 or curr_bbw_rank > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries in low volatility regime
            # Regime filter: BBW < 20th percentile = low volatility squeeze
            low_vol_regime = curr_bbw_rank < 20
            
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirm = curr_volume > 1.3 * curr_vol_ma
            
            # Long entry: RSI(2) oversold (< 10) in uptrend (price > 1d EMA50) + low vol regime
            if low_vol_regime and vol_confirm and curr_close > curr_ema50_1d:
                if curr_rsi < 10:
                    signals[i] = 0.25
                    position = 1
            # Short entry: RSI(2) overbought (> 90) in downtrend (price < 1d EMA50) + low vol regime
            elif low_vol_regime and vol_confirm and curr_close < curr_ema50_1d:
                if curr_rsi > 90:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals