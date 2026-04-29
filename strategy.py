#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 12h RSI mean reversion
# In low volatility (BBW < 20th percentile), use 12h RSI for mean reversion:
#   Long when RSI < 30, Short when RSI > 70
# In high volatility (BBW > 80th percentile), follow 12h EMA50 trend:
#   Long when price > EMA50, Short when price < EMA50
# This adapts to market regimes: mean revert in ranging, trend follow in volatile
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_BBW_Regime_RSI_EMA_Adaptive_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h RSI(14) for mean reversion signals
    delta = pd.Series(close_12h).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.fillna(50).values  # neutral when undefined
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 12h EMA50 for trend following
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h Bollinger Band Width (20,2) for regime detection
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle * 100  # as percentage
    
    # Regime thresholds: 20th and 80th percentiles of BBW (using expanding window)
    bbw_pct_20 = pd.Series(bb_width).expanding(min_periods=50).quantile(0.20).values
    bbw_pct_80 = pd.Series(bb_width).expanding(min_periods=50).quantile(0.80).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # max(20, 14, 50) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width[i]) or 
            np.isnan(bbw_pct_20[i]) or 
            np.isnan(bbw_pct_80[i]) or
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bbw = bb_width[i]
        curr_bbw_20 = bbw_pct_20[i]
        curr_bbw_80 = bbw_pct_80[i]
        curr_rsi = rsi_12h_aligned[i]
        curr_ema = ema_50_12h_aligned[i]
        
        # Determine regime based on BBW
        is_low_vol = curr_bbw < curr_bbw_20   # ranging market
        is_high_vol = curr_bbw > curr_bbw_80  # trending market
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions depend on regime
            if is_low_vol:
                # In ranging: exit when RSI reverts to 50
                if curr_rsi >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # high_vol or neutral
                # In trending: exit when price crosses below EMA50
                if curr_close < curr_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions depend on regime
            if is_low_vol:
                # In ranging: exit when RSI reverts to 50
                if curr_rsi <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # high_vol or neutral
                # In trending: exit when price crosses above EMA50
                if curr_close > curr_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
                    
        else:  # Flat - look for new entries
            if is_low_vol:
                # Ranging market: mean reversion with RSI extremes
                if curr_rsi < 30:
                    signals[i] = 0.25
                    position = 1
                elif curr_rsi > 70:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_high_vol:
                # Trending market: follow EMA50
                if curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                elif curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime: no clear signal
                signals[i] = 0.0
    
    return signals