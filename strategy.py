#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume spike filter
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction and strength.
# Alligator lines converging (chop) = no trade; diverging (trend) = trade in direction of mouth.
# 1d EMA34 filters for higher timeframe trend alignment.
# Volume spike confirms momentum behind breakout.
# ATR-based stoploss manages risk. Works in bull/bear via trend filter.
# Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime filter (avoid high volatility chop)
    atr_percentile = pd.Series(atr).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 20 else np.nan, raw=True
    ).values
    vol_regime_filter = atr <= atr_percentile  # Only trade in low/medium volatility regimes
    
    # Williams Alligator: SMMA (Smoothed Moving Average) with different periods
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: JAW (13, 8), TEETH (8, 5), LIPS (5, 3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator signals: 
    # - All lines intertwined (chop): no trade
    # - Lines diverging upward (JAW > TEETH > LIPS): uptrend
    # - Lines diverging downward (JAW < TEETH < LIPS): downtrend
    alligator_long = (jaws > teeth) & (teeth > lips) & (~np.isnan(jaws))
    alligator_short = (jaws < teeth) & (teeth < lips) & (~np.isnan(jaws))
    alligator_chop = ~(alligator_long | alligator_short) & (~np.isnan(jaws))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    
    start_idx = max(50, 34, 20, 14, 13)  # warmup for EMA34, Alligator, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_regime = vol_regime_filter[i]
        curr_alligator_long = alligator_long[i]
        curr_alligator_short = alligator_short[i]
        curr_alligator_chop = alligator_chop[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Alligator signals chop or reverse
            # 3. Price crosses below 1d EMA34 (trend change)
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_low <= stop_price or
                curr_alligator_chop or
                curr_alligator_short or
                curr_close < curr_ema_34_1d or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Alligator signals chop or reverse
            # 3. Price crosses above 1d EMA34 (trend change)
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_high >= stop_price or
                curr_alligator_chop or
                curr_alligator_long or
                curr_close > curr_ema_34_1d or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter in low/medium volatility regimes to avoid whipsaws
            if not curr_vol_regime:
                signals[i] = 0.0
                continue
                
            # Only trade when Alligator is trending (not chop)
            if curr_alligator_chop:
                signals[i] = 0.0
                continue
                
            # Long entry: Alligator uptrend + above 1d EMA34 + volume confirm
            if (curr_alligator_long and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: Alligator downtrend + below 1d EMA34 + volume confirm
            elif (curr_alligator_short and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume spike filter
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction and strength.
# Alligator lines converging (chop) = no trade; diverging (trend) = trade in direction of mouth.
# 1d EMA34 filters for higher timeframe trend alignment.
# Volume spike confirms momentum behind breakout.
# ATR-based stoploss manages risk. Works in bull/bear via trend filter.
# Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR percentile for volatility regime filter (avoid high volatility chop)
    atr_percentile = pd.Series(atr).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 20 else np.nan, raw=True
    ).values
    vol_regime_filter = atr <= atr_percentile  # Only trade in low/medium volatility regimes
    
    # Williams Alligator: SMMA (Smoothed Moving Average) with different periods
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: JAW (13, 8), TEETH (8, 5), LIPS (5, 3)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator signals: 
    # - All lines intertwined (chop): no trade
    # - Lines diverging upward (JAW > TEETH > LIPS): uptrend
    # - Lines diverging downward (JAW < TEETH < LIPS): downtrend
    alligator_long = (jaw > teeth) & (teeth > lips) & (~np.isnan(jaw))
    alligator_short = (jaw < teeth) & (teeth < lips) & (~np.isnan(jaw))
    alligator_chop = ~(alligator_long | alligator_short) & (~np.isnan(jaw))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0  # For trailing stop
    min_low_since_entry = 0.0   # For trailing stop
    
    start_idx = max(50, 34, 20, 14, 13)  # warmup for EMA34, Alligator, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_vol_regime = vol_regime_filter[i]
        curr_alligator_long = alligator_long[i]
        curr_alligator_short = alligator_short[i]
        curr_alligator_chop = alligator_chop[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR below entry
            fixed_stop = entry_price - 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = max(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Alligator signals chop or reverse
            # 3. Price crosses below 1d EMA34 (trend change)
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_low <= stop_price or
                curr_alligator_chop or
                curr_alligator_short or
                curr_close < curr_ema_34_1d or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # Dynamic stoploss: ATR-based trailing stop
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            # Fixed stoploss: 2.0 * ATR above entry
            fixed_stop = entry_price + 2.0 * atr_at_entry
            # Use the tighter of the two stops
            stop_price = min(trail_stop, fixed_stop)
            
            # Exit conditions:
            # 1. Stoploss hit (trailing or fixed)
            # 2. Alligator signals chop or reverse
            # 3. Price crosses above 1d EMA34 (trend change)
            # 4. Volatility regime shifts to high (avoid chop)
            if (curr_high >= stop_price or
                curr_alligator_chop or
                curr_alligator_long or
                curr_close > curr_ema_34_1d or
                not curr_vol_regime):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter in low/medium volatility regimes to avoid whipsaws
            if not curr_vol_regime:
                signals[i] = 0.0
                continue
                
            # Only trade when Alligator is trending (not chop)
            if curr_alligator_chop:
                signals[i] = 0.0
                continue
                
            # Long entry: Alligator uptrend + above 1d EMA34 + volume confirm
            if (curr_alligator_long and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: Alligator downtrend + below 1d EMA34 + volume confirm
            elif (curr_alligator_short and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals