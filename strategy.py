#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Fractal breakout with 12h trend filter
# - Elder Ray Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Williams Fractal identifies swing points: bullish fractal = low < two left/right lows
# - Long when Bull Power > 0 AND bullish fractal confirmed AND 12h close > EMA50
# - Short when Bear Power > 0 AND bearish fractal confirmed AND 12h close < EMA50
# - Exit when Elder Ray power reverses OR price retreats to EMA13
# - Uses 12h trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Works in bull via trend continuation, in bear via fade at extremes

name = "6h_12h_elder_ray_fractal_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Elder Ray components: EMA13
    close_s = prices['close']
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'] - ema13
    bear_power = ema13 - prices['low']
    
    # Pre-compute Williams Fractals (5-bar: 2 left, center, 2 right)
    high = prices['high'].values
    low = prices['low'].values
    bullish_fractal = np.full(n, np.nan)
    bearish_fractal = np.full(n, np.nan)
    
    # Calculate fractals: need 2 bars on each side
    for i in range(2, n-2):
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]  # bullish fractal at low
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]  # bearish fractal at high
    
    # Pre-compute volume confirmation: > 1.5x 20-bar average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for dynamic stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 12h data
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    h_12h_aligned = align_htf_to_ltf(prices, df_12h, h_12h)
    l_12h_aligned = align_htf_to_ltf(prices, df_12h, l_12h)
    c_12h_aligned = align_htf_to_ltf(prices, df_12h, c_12h)
    
    # Pre-compute 12h EMA(50) for trend filter
    ema50_12h = pd.Series(c_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute 12h EMA13 for Elder Ray HTF (for additional filter)
    ema13_12h = pd.Series(c_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema13_12h_aligned[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Check for confirmed fractals (need 2 bars to the right for confirmation)
        bullish_confirm = not np.isnan(bullish_fractal[i-2]) if i >= 2 else False
        bearish_confirm = not np.isnan(bearish_fractal[i-2]) if i >= 2 else False
        
        if position == 0:  # Flat - look for new entries
            # Long conditions:
            # 1. Bull Power > 0 (strong bullish momentum)
            # 2. Confirmed bullish fractal (recent swing low)
            # 3. Volume spike (confirmation of participation)
            # 4. 12h trend filter: price > EMA50 AND price > EMA13 (bullish regime)
            if (bull_power[i] > 0 and 
                bullish_confirm and 
                vol_spike.iloc[i] and
                prices['close'].iloc[i] > ema50_12h_aligned[i] and
                prices['close'].iloc[i] > ema13_12h_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short conditions:
            # 1. Bear Power > 0 (strong bearish momentum)
            # 2. Confirmed bearish fractal (recent swing high)
            # 3. Volume spike
            # 4. 12h trend filter: price < EMA50 AND price < EMA13 (bearish regime)
            elif (bear_power[i] > 0 and 
                  bearish_confirm and 
                  vol_spike.iloc[i] and
                  prices['close'].iloc[i] < ema50_12h_aligned[i] and
                  prices['close'].iloc[i] < ema13_12h_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
        
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Ray power reverses (loss of momentum)
            # 2. Price retraces to EMA13 (dynamic support/resistance)
            # 3. ATR-based stoploss (2.5x ATR)
            exit_signal = False
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or  # Bull Power turned negative
                    prices['close'].iloc[i] <= ema13[i] or  # Price back to EMA13
                    prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):  # ATR stop
                    exit_signal = True
            elif position == -1:  # Short position
                if (bear_power[i] <= 0 or  # Bear Power turned negative
                    prices['close'].iloc[i] >= ema13[i] or  # Price back to EMA13
                    prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):  # ATR stop
                    exit_signal = True
            
            if exit_signal:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals