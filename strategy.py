#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 12h RSI Divergence Filter
# Uses Bollinger Band Width percentile to identify ranging (BW < 30th percentile) vs trending (BW > 70th percentile) regimes.
# In ranging markets: mean reversion at Bollinger Bands (buy at lower band, sell at upper band).
# In trending markets: trend continuation with 12h RSI divergence filter (avoid counter-trend entries).
# Designed to work in both bull (trending) and bear (ranging) markets with low trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # Bollinger Band Width percentile rank (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 12-hour RSI for trend filter and divergence detection
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 12-hour RSI slope for divergence detection (3-period slope)
    rsi_slope = np.gradient(rsi_12h_aligned)
    rsi_slope_smooth = pd.Series(rsi_slope).rolling(window=3, min_periods=3).mean().values
    
    # Price position within Bollinger Bands (0 = lower band, 1 = upper band)
    bb_position = (close - lower_band) / (upper_band - lower_band)
    bb_position = np.clip(bb_position, 0, 1)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(rsi_slope_smooth[i]) or np.isnan(bb_position[i]) or
            np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            continue
        
        bb_percent = bb_width_percentile[i]
        rsi = rsi_12h_aligned[i]
        rsi_slope_val = rsi_slope_smooth[i]
        price_pos = bb_position[i]
        
        # Regime definition:
        # Ranging: BB Width < 30th percentile (low volatility, mean revert)
        # Trending: BB Width > 70th percentile (high volatility, trend follow)
        is_ranging = bb_percent < 30
        is_trending = bb_percent > 70
        
        if is_ranging:
            # Mean reversion in ranging markets
            # Buy near lower band, sell near upper band
            if price_pos <= 0.15 and rsi < 40:  # Oversold near lower band
                signals[i] = 0.25
            elif price_pos >= 0.85 and rsi > 60:  # Overbought near upper band
                signals[i] = -0.25
            # Exit when price returns to middle
            elif (i > 0 and 
                  ((signals[i-1] == 0.25 and price_pos >= 0.5) or
                   (signals[i-1] == -0.25 and price_pos <= 0.5))):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
                
        elif is_trending:
            # Trend following with RSI divergence filter in trending markets
            # Long: Uptrend with no bearish divergence
            if (rsi > 50 and rsi_slope_val > 0 and 
                close[i] > sma_20[i] and 
                not (rsi > 60 and rsi_slope_val < 0)):  # No bearish divergence
                signals[i] = 0.25
            # Short: Downtrend with no bullish divergence
            elif (rsi < 50 and rsi_slope_val < 0 and 
                  close[i] < sma_20[i] and
                  not (rsi < 40 and rsi_slope_val > 0)):  # No bullish divergence
                signals[i] = -0.25
            # Exit when trend weakens or RSI reverses
            elif (i > 0 and 
                  ((signals[i-1] == 0.25 and (rsi <= 50 or rsi_slope_val <= 0 or close[i] <= sma_20[i])) or
                   (signals[i-1] == -0.25 and (rsi >= 50 or rsi_slope_val >= 0 or close[i] >= sma_20[i])))):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            # Transition regime - hold or exit
            if i > 0:
                signals[i] = signals[i-1]
                # Exit positions during transition
                if (signals[i-1] == 0.25 and price_pos >= 0.5) or \
                   (signals[i-1] == -0.25 and price_pos <= 0.5):
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_BBW_RSI_Divergence_Regime"
timeframe = "6h"
leverage = 1.0