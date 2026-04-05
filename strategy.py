#!/usr/bin/env python3
"""
exp_7251_6h_adx_elder_ray_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ADX regime filter and volume confirmation.
In strong trends (ADX>25): trade in direction of Elder Ray (Bull/Bear Power).
In weak trends (ADX<=25): fade extreme Elder Ray readings with volume confirmation.
Uses 1d EMA200 as additional trend filter to avoid counter-trend trades in strong regimes.
Designed for 6h timeframe to capture multi-day swings with ~12-37 trades/year (50-150 total over 4 years).
Works in bull markets via trend continuation and bear markets via mean reversion in ranges.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7251_6h_adx_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 25
EMA_FILTER_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_CONFIRM_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_FILTER_PERIOD, adjust=False, min_periods=EMA_FILTER_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX calculation
    plus_dm = pd.Series(np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                                 np.maximum(high - np.roll(high, 1), 0), 0))
    minus_dm = pd.Series(np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                                  np.maximum(np.roll(low, 1) - low, 0), 0))
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_tr = tr.ewm(span=1, adjust=False).mean()  # True Range
    
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / 
                     atr_tr.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean())
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / 
                      atr_tr.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # ATR for stoploss
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, ADX_PERIOD, EMA_FILTER_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_CONFIRM_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend regime
        strong_trend = adx[i] > ADX_THRESHOLD
        weak_trend = adx[i] <= ADX_THRESHOLD
        
        # Price relative to 1d EMA200 filter
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        # Entry logic
        if position == 0:
            # Strong trend: trade with Elder Ray direction
            if strong_trend:
                if bull_power[i] > 0 and price_above_ema and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power[i] < 0 and price_below_ema and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            # Weak trend: fade extreme Elder Ray readings
            else:
                if bull_power[i] < 0 and price_below_ema and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power[i] > 0 and price_above_ema and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7251_6h_adx_elder_ray_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with ADX regime filter and volume confirmation.
In strong trends (ADX>25): trade in direction of Elder Ray (Bull/Bear Power).
In weak trends (ADX<=25): fade extreme Elder Ray readings with volume confirmation.
Uses 1d EMA200 as additional trend filter to avoid counter-trend trades in strong regimes.
Designed for 6h timeframe to capture multi-day swings with ~12-37 trades/year (50-150 total over 4 years).
Works in bull markets via trend continuation and bear markets via mean reversion in ranges.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7251_6h_adx_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 25
EMA_FILTER_PERIOD = 200
VOL_MA_PERIOD = 20
VOL_CONFIRM_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 12  # ~3 days

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_FILTER_PERIOD, adjust=False, min_periods=EMA_FILTER_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX calculation
    plus_dm = pd.Series(np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                                 np.maximum(high - np.roll(high, 1), 0), 0))
    minus_dm = pd.Series(np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                                  np.maximum(np.roll(low, 1) - low, 0), 0))
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_tr = tr.ewm(span=1, adjust=False).mean()  # True Range
    
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / 
                     atr_tr.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean())
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean() / 
                      atr_tr.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # ATR for stoploss
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, ADX_PERIOD, EMA_FILTER_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]):
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_CONFIRM_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend regime
        strong_trend = adx[i] > ADX_THRESHOLD
        weak_trend = adx[i] <= ADX_THRESHOLD
        
        # Price relative to 1d EMA200 filter
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        # Entry logic
        if position == 0:
            # Strong trend: trade with Elder Ray direction
            if strong_trend:
                if bull_power[i] > 0 and price_above_ema and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power[i] < 0 and price_below_ema and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            # Weak trend: fade extreme Elder Ray readings
            else:
                if bull_power[i] < 0 and price_below_ema and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power[i] > 0 and price_above_ema and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals