# 4h_Donchian_Breakout_Volume_Trend_Regime_v1
# 4h strategy combining Donchian breakout with volume confirmation, trend filter, and Chop regime filter.
# Long: Price breaks above Donchian(20) high + volume > 1.5x 20-period avg + EMA20 > EMA50 + Chop > 61.8 (range)
# Short: Price breaks below Donchian(20) low + volume > 1.5x 20-period avg + EMA20 < EMA50 + Chop > 61.8 (range)
# Exit: Opposite breakout or trend reversal or Chop < 38.2 (trend)
# Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
# Works in bull markets (breakout continuation) and bear markets (avoids false breakouts via Chop filter)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppy market indicator (Chop) - 14-period
    # Chop = 100 * log10(sum(ATR(1) over n) / (max(high-n) - min(low-n))) / log10(n)
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first TR
    atr = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) is just TR
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_val = max_high - min_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    chop = 100 * (np.log10(sum_atr / range_val) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50 and Chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_20[i] > ema_50[i]
        downtrend = ema_20[i] < ema_50[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime filter (only trade in ranging markets)
        chop_filter = chop[i] > 61.8  # range when Chop > 61.8
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakdown_down = close[i] < donchian_low[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above Donchian high + Chop filter
            if uptrend and vol_confirm and breakout_up and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below Donchian low + Chop filter
            elif downtrend and vol_confirm and breakdown_down and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal, volume breakdown, Chop trend signal, or opposite breakout
            if (not uptrend) or (not vol_confirm) or (chop[i] < 38.2) or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal, volume breakdown, Chop trend signal, or opposite breakout
            if (not downtrend) or (not vol_confirm) or (chop[i] < 38.2) or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Regime_v1"
timeframe = "4h"
leverage = 1.0