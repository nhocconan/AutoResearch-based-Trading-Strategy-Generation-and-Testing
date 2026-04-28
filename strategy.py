#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility normalization and filtering
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d RSI(14) for overbought/oversold conditions
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # 1d Bollinger Bands (20, 2) for volatility regime
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Align 1d indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # 6h ATR(14) for position sizing volatility adjustment
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    atr_14_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(bb_width_1d_aligned[i]) or 
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: avoid extremely low or high volatility
        vol_normal = (bb_width_1d_aligned[i] > 0.01) & (bb_width_1d_aligned[i] < 0.05)
        
        # Trend filter: price relative to 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Momentum filter: RSI not extreme
        momentum_ok = (rsi_14_1d_aligned[i] > 30) & (rsi_14_1d_aligned[i] < 70)
        
        # Volatility-adjusted position size
        vol_factor = np.clip(atr_14_6h[i] / (0.01 * close[i]), 0.5, 2.0)
        base_size = 0.25
        position_size = base_size * vol_factor
        
        # Entry conditions
        # Long: uptrend + normal volatility + RSI not oversold
        long_entry = trend_up & vol_normal & momentum_ok
        # Short: downtrend + normal volatility + RSI not overbought
        short_entry = trend_down & vol_normal & momentum_ok
        
        # Exit conditions: trend reversal or volatility extreme
        trend_reversal = (~trend_up & position == 1) | (~trend_down & position == -1)
        vol_extreme = bb_width_1d_aligned[i] >= 0.05
        
        if long_entry and position <= 0:
            signals[i] = position_size
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -position_size
            position = -1
        elif trend_reversal or vol_extreme:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_VolatilityRegime_Trend_Momentum"
timeframe = "6h"
leverage = 1.0