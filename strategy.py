#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) in bullish regime (price > 1d EMA34) with volume spike
# Short when Bear Power < 0 (close < EMA13) AND Bull Power < 0 (close < EMA13) in bearish regime (price < 1d EMA34) with volume spike
# Uses 1d EMA34 for trend filter to avoid counter-trend whipsaws
# Volume confirmation ensures moves have institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 6h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: EMA13 for reference
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = Close - EMA13 (same calculation, interpreted differently)
    bear_power = close - ema13
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for EMA34 and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(ema13[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 AND Bear Power < 0 in bullish regime
                # (close > EMA13 AND close < EMA13 is impossible, so we use:
                # Bull Power > 0 means close > EMA13)
                if is_bullish_regime and curr_bull_power > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bull Power < 0 AND Bear Power < 0 in bearish regime
                # (close < EMA13 means both powers negative)
                elif is_bearish_regime and curr_bull_power < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Bull Power turns negative (close < EMA13) OR regime changes to bearish
            if curr_bull_power <= 0 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Bull Power turns positive (close > EMA13) OR regime changes to bullish
            if curr_bull_power >= 0 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals