# 6h_OrderBlock_Retest_LiquidityImbalance
# Hypothesis: Institutional order blocks form at key support/resistance levels. Price retesting these blocks with liquidity imbalance (volume spike + price rejection) provides high-probability entries. Works in both bull/bear as it identifies institutional participation.
# Uses 1d order blocks identified by volume profile + price action. Entry on 6h retest with volume confirmation. Stop via signal reversal.

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
    
    # Calculate 1d order blocks using volume profile and price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Identify 1d order blocks: high volume nodes with strong close
    vol_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Volume-weighted average price approximation for each day
    vwap_1d = (high_1d + low_1d + close_1d) / 3
    
    # Identify high volume areas (top 30% volume days)
    vol_threshold = np.percentile(vol_1d, 70)
    high_vol_mask = vol_1d >= vol_threshold
    
    # Bullish OB: high volume day with close > vwap
    # Bearish OB: high volume day with close < vwap
    bullish_ob = high_vol_mask & (close_1d > vwap_1d)
    bearish_ob = high_vol_mask & (close_1d < vwap_1d)
    
    # Create OB levels (using the vwap of those days)
    bullish_ob_levels = np.where(bullish_ob, vwap_1d, np.nan)
    bearish_ob_levels = np.where(bearish_ob, vwap_1d, np.nan)
    
    # Forward fill to get active OB levels
    bullish_ob_series = pd.Series(bullish_ob_levels)
    bearish_ob_series = pd.Series(bearish_ob_levels)
    bullish_ob_filled = bullish_ob_series.ffill().bfill().values
    bearish_ob_filled = bearish_ob_series.ffill().bfill().values
    
    # Align OB levels to 6h timeframe
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_filled)
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_filled)
    
    # Volume spike detector (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price action rejection signals
    # Bullish rejection: long lower wick, close near high
    lower_wick = close - low
    upper_wick = high - close
    body = np.abs(close - open_) if 'open' in prices.columns else np.abs(close - np.roll(close, 1))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    body = np.abs(close - open_)
    body = np.where(body == 0, 0.001, body)  # avoid division by zero
    
    bullish_rejection = (lower_wick > 2 * body) & (close > (high - 0.3 * (high - low)))
    bearish_rejection = (upper_wick > 2 * body) & (close < (low + 0.3 * (high - low)))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20)  # volume MA20, need price data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bullish_ob_aligned[i]) or 
            np.isnan(bearish_ob_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Distance to OB levels (avoid division by zero)
        dist_to_bull_ob = np.abs(close[i] - bullish_ob_aligned[i]) / close[i] if not np.isnan(bullish_ob_aligned[i]) else 1.0
        dist_to_bear_ob = np.abs(close[i] - bearish_ob_aligned[i]) / close[i] if not np.isnan(bearish_ob_aligned[i]) else 1.0
        
        # Near OB level (within 0.5%)
        near_bull_ob = dist_to_bull_ob < 0.005
        near_bear_ob = dist_to_bear_ob < 0.005
        
        if position == 0:
            # Long: near bullish OB + bullish rejection + volume spike
            if near_bull_ob and bullish_rejection[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: near bearish OB + bearish rejection + volume spike
            elif near_bear_ob and bearish_rejection[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: near bearish OB with bearish rejection OR opposite signal
            if near_bear_ob and bearish_rejection[i] and volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: near bullish OB with bullish rejection OR opposite signal
            if near_bull_ob and bullish_rejection[i] and volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_OrderBlock_Retest_LiquidityImbalance"
timeframe = "6h"
leverage = 1.0