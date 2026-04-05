#!/usr/bin/env python3
"""
exp_6999_6h_elder_ray_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX + EMA200).
In strong trends (ADX>25): trade Elder Ray extremes in trend direction.
In weak trends (ADX<20): fade Elder Ray extremes (mean reversion).
Uses 1d EMA200 for long-term bias to avoid counter-trend trades.
Designed for 6h timeframe with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to volatility regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6999_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13
EMA200_PERIOD = 200
ADX_PERIOD = 14
ADX_STRONG = 25
ADX_WEAK = 20
ELDER_THRESHOLD = 0.02  # 2% of price for extreme reading
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 40  # ~10 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 220:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for trend bias
    ema200_1d = pd.Series(close_1d).ewm(span=EMA200_PERIOD, adjust=False, min_periods=EMA200_PERIOD).mean().values
    
    # 1d ADX for regime detection
    # +DI
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    # -DI
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align to LTF (6h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=EMA13_PERIOD, adjust=False, min_periods=EMA13_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA13_PERIOD, EMA200_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine regime
        strong_trend = adx_1d_aligned[i] > ADX_STRONG
        weak_trend = adx_1d_aligned[i] < ADX_WEAK
        
        # Long-term bias
        long_bias = close[i] > ema200_1d_aligned[i]
        short_bias = close[i] < ema200_1d_aligned[i]
        
        # Elder Ray signals
        bull_extreme = bull_power[i] > ELDER_THRESHOLD * close[i]
        bear_extreme = bear_power[i] < -ELDER_THRESHOLD * close[i]
        
        # Entry logic based on regime
        if position == 0:
            if strong_trend:
                # Trend following: trade with the trend
                if bull_extreme and long_bias:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_extreme and short_bias:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            elif weak_trend:
                # Mean reversion: fade extremes
                if bull_extreme and short_bias:  # overbought in downtrend bias
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_extreme and long_bias:  # oversold in uptrend bias
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trades
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6999_6h_elder_ray_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX + EMA200).
In strong trends (ADX>25): trade Elder Ray extremes in trend direction.
In weak trends (ADX<20): fade Elder Ray extremes (mean reversion).
Uses 1d EMA200 for long-term bias to avoid counter-trend trades.
Designed for 6h timeframe with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to volatility regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6999_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA13_PERIOD = 13
EMA200_PERIOD = 200
ADX_PERIOD = 14
ADX_STRONG = 25
ADX_WEAK = 20
ELDER_THRESHOLD = 0.02  # 2% of price for extreme reading
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 40  # ~10 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 220:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for trend bias
    ema200_1d = pd.Series(close_1d).ewm(span=EMA200_PERIOD, adjust=False, min_periods=EMA200_PERIOD).mean().values
    
    # 1d ADX for regime detection
    # +DI
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    # -DI
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align to LTF (6h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=EMA13_PERIOD, adjust=False, min_periods=EMA13_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA13_PERIOD, EMA200_PERIOD, ADX_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine regime
        strong_trend = adx_1d_aligned[i] > ADX_STRONG
        weak_trend = adx_1d_aligned[i] < ADX_WEAK
        
        # Long-term bias
        long_bias = close[i] > ema200_1d_aligned[i]
        short_bias = close[i] < ema200_1d_aligned[i]
        
        # Elder Ray signals
        bull_extreme = bull_power[i] > ELDER_THRESHOLD * close[i]
        bear_extreme = bear_power[i] < -ELDER_THRESHOLD * close[i]
        
        # Entry logic based on regime
        if position == 0:
            if strong_trend:
                # Trend following: trade with the trend
                if bull_extreme and long_bias:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_extreme and short_bias:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            elif weak_trend:
                # Mean reversion: fade extremes
                if bull_extreme and short_bias:  # overbought in downtrend bias
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_extreme and long_bias:  # oversold in uptrend bias
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trades
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals