#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze Breakout with 1d EMA50 trend and volume confirmation.
# The Bollinger Band Squeeze (low volatility followed by expansion) captures explosive moves.
# Trade only when BB width is below its 50-period percentile (squeeze) and price breaks out
# of the bands with volume confirmation. 1d EMA50 filters for higher timeframe trend direction.
# This works in both bull and bear markets by catching volatility expansions in the direction
# of the higher timeframe trend. Target: 15-25 trades per year to minimize fee drag.

name = "12h_BollingerSqueeze_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA50 for trend direction ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Bollinger Bands on 12h (20, 2) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Squeeze: width below 20th percentile of lookback
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = close[i]
        sma_val = sma_20[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        bb_width_val = bb_width[i]
        bb_width_pct = bb_width_percentile[i]
        ema_val = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(sma_val) or np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(bb_width_val) or np.isnan(bb_width_pct) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band Squeeze (low volatility) + breakout in trend direction
            # Squeeze: BB width below 20th percentile of lookback
            # Long: price breaks above upper band in uptrend (close > EMA50)
            # Short: price breaks below lower band in downtrend (close < EMA50)
            if bb_width_pct < 0.2:  # Squeeze condition
                if close_val > upper_val and close_val > ema_val:  # Long breakout
                    if vol_ratio_val > 1.5:  # Volume confirmation
                        signals[i] = 0.25
                        position = 1
                elif close_val < lower_val and close_val < ema_val:  # Short breakout
                    if vol_ratio_val > 1.5:  # Volume confirmation
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band (mean reversion)
            # or trend reversal (price closes below EMA50)
            if close_val < sma_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band (mean reversion)
            # or trend reversal (price closes above EMA50)
            if close_val > sma_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals