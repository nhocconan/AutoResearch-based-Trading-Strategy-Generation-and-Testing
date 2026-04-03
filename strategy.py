#!/usr/bin/env python3
"""
Experiment #307: 6h Camarilla Pivot + 1d Volume Spike + Weekly Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d volume spikes (>2.0x average) and 1d weekly trend (price > weekly EMA20) 
capture high-probability mean reversion and breakout moves. Weekly trend filter adapts 
to bull/bear regimes: in bull markets, favor long breakouts at R4; in bear markets, 
favor short breakdowns at S4. Mean reversion at R3/S3 works in ranging markets. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. 
Volume confirmation ensures institutional participation. Weekly EMA20 provides smooth 
trend filter less prone to whipsaw than shorter EMAs.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- Weekly EMA20 calculated on 1d data then aligned to 6h
- Camarilla levels calculated from previous 1d OHLC
- Exits on Camarilla H3/L3 reversion (mean reversion) or opposite pivot touch (breakout)
- ATR-based stoploss for risk management
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_307_6h_camarilla_1d_volume_weekly_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots, volume MA, weekly EMA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    #            H2 = close + 0.75*(high-low),  L2 = close - 0.75*(high-low)
    #            H1 = close + 0.5*(high-low),   L1 = close - 0.5*(high-low)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC (aligned to current 6h bar)
        phigh = df_1d['high'].values[i-1] if i-1 < len(df_1d['high'].values) else np.nan
        plow = df_1d['low'].values[i-1] if i-1 < len(df_1d['low'].values) else np.nan
        pclose = df_1d['close'].values[i-1] if i-1 < len(df_1d['close'].values) else np.nan
        
        if not (np.isnan(phigh) or np.isnan(plow) or np.isnan(pclose)):
            diff = phigh - plow
            camarilla_h4[i] = pclose + 1.5 * diff
            camarilla_l4[i] = pclose - 1.5 * diff
            camarilla_h3[i] = pclose + 1.125 * diff
            camarilla_l3[i] = pclose - 1.125 * diff
            camarilla_h2[i] = pclose + 0.75 * diff
            camarilla_l2[i] = pclose - 0.75 * diff
            camarilla_h1[i] = pclose + 0.5 * diff
            camarilla_l1[i] = pclose - 0.5 * diff
            camarilla_close[i] = pclose
        else:
            # Propagate previous values for warmup
            camarilla_h4[i] = camarilla_h4[i-1] if i > 0 else np.nan
            camarilla_l4[i] = camarilla_l4[i-1] if i > 0 else np.nan
            camarilla_h3[i] = camarilla_h3[i-1] if i > 0 else np.nan
            camarilla_l3[i] = camarilla_l3[i-1] if i > 0 else np.nan
            camarilla_h2[i] = camarilla_h2[i-1] if i > 0 else np.nan
            camarilla_l2[i] = camarilla_l2[i-1] if i > 0 else np.nan
            camarilla_h1[i] = camarilla_h1[i-1] if i > 0 else np.nan
            camarilla_l1[i] = camarilla_l1[i-1] if i > 0 else np.nan
            camarilla_close[i] = camarilla_close[i-1] if i > 0 else np.nan
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros(len(df_1d['volume'].values))
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_20[20:]
    vol_ratio_1d[:20] = 1.0
    
    # Align volume ratio to 6h
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate weekly EMA20 on 1d data (using Friday's close as weekly proxy)
    # Simplified: EMA20 of 1d close
    weekly_ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema_20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    bars_since_entry = 0
    
    warmup = 100  # Warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_aligned[i] > 2.0
        
        # --- Weekly Trend Filter ---
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # --- Camarilla Levels (using previous bar's levels to avoid look-ahead) ---
        # Use levels from previous bar (already calculated from previous 1d bar)
        h4 = camarilla_h4[i-1] if i > 0 else np.nan
        l4 = camarilla_l4[i-1] if i > 0 else np.nan
        h3 = camarilla_h3[i-1] if i > 0 else np.nan
        l3 = camarilla_l3[i-1] if i > 0 else np.nan
        h2 = camarilla_h2[i-1] if i > 0 else np.nan
        l2 = camarilla_l2[i-1] if i > 0 else np.nan
        h1 = camarilla_h1[i-1] if i > 0 else np.nan
        l1 = camarilla_l1[i-1] if i > 0 else np.nan
        pivot = camarilla_close[i-1] if i > 0 else np.nan
        
        if np.isnan(h4) or np.isnan(l4) or np.isnan(h3) or np.isnan(l3):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions based on entry type
            if position_side > 0:  # Long
                # Mean reversion long: exit at H3 (take profit)
                if not weekly_uptrend and close[i] >= h3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Breakout long: exit if price revisits pivot (failed breakout)
                elif weekly_uptrend and close[i] <= pivot:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Mean reversion short: exit at L3 (take profit)
                if not weekly_downtrend and close[i] <= l3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Breakout short: exit if price revisits pivot (failed breakout)
                elif weekly_downtrend and close[i] >= pivot:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean Reversion Long: Price touches S3/S4 + weekly downtrend (bear market bounce) + volume spike
        mean_rev_long = ((close[i] <= l3 or close[i] <= l4) and 
                         weekly_downtrend and volume_spike)
        
        # Mean Reversion Short: Price touches R3/R4 + weekly uptrend (bull market pullback) + volume spike
        mean_rev_short = ((close[i] >= h3 or close[i] >= h4) and 
                          weekly_uptrend and volume_spike)
        
        # Breakout Long: Price breaks above H4 + weekly uptrend + volume spike
        breakout_long = (close[i] > h4 and weekly_uptrend and volume_spike)
        
        # Breakout Short: Price breaks below L4 + weekly downtrend + volume spike
        breakout_short = (close[i] < l4 and weekly_downtrend and volume_spike)
        
        if mean_rev_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            bars_since_entry = 0
            signals[i] = SIZE
        elif mean_rev_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #307: 6h Camarilla Pivot + 1d Volume Spike + Weekly Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d volume spikes (>2.0x average) and 1d weekly trend (price > weekly EMA20) 
capture high-probability mean reversion and breakout moves. Weekly trend filter adapts 
to bull/bear regimes: in bull markets, favor long breakouts at R4; in bear markets, 
favor short breakdowns at S4. Mean reversion at R3/S3 works in ranging markets. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. 
Volume confirmation ensures institutional participation. Weekly EMA20 provides smooth 
trend filter less prone to whipsaw than shorter EMAs.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- Weekly EMA20 calculated on 1d data then aligned to 6h
- Camarilla levels calculated from previous 1d OHLC
- Exits on Camarilla H3/L3 reversion (mean reversion) or opposite pivot touch (breakout)
- ATR-based stoploss for risk management
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_307_6h_camarilla_1d_volume_weekly_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots, volume MA, weekly EMA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    #            H2 = close + 0.75*(high-low),  L2 = close - 0.75*(high-low)
    #            H1 = close + 0.5*(high-low),   L1 = close - 0.5*(high-low)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC (aligned to current 6h bar)
        phigh = df_1d['high'].values[i-1] if i-1 < len(df_1d['high'].values) else np.nan
        plow = df_1d['low'].values[i-1] if i-1 < len(df_1d['low'].values) else np.nan
        pclose = df_1d['close'].values[i-1] if i-1 < len(df_1d['close'].values) else np.nan
        
        if not (np.isnan(phigh) or np.isnan(plow) or np.isnan(pclose)):
            diff = phigh - plow
            camarilla_h4[i] = pclose + 1.5 * diff
            camarilla_l4[i] = pclose - 1.5 * diff
            camarilla_h3[i] = pclose + 1.125 * diff
            camarilla_l3[i] = pclose - 1.125 * diff
            camarilla_h2[i] = pclose + 0.75 * diff
            camarilla_l2[i] = pclose - 0.75 * diff
            camarilla_h1[i] = pclose + 0.5 * diff
            camarilla_l1[i] = pclose - 0.5 * diff
            camarilla_close[i] = pclose
        else:
            # Propagate previous values for warmup
            camarilla_h4[i] = camarilla_h4[i-1] if i > 0 else np.nan
            camarilla_l4[i] = camarilla_l4[i-1] if i > 0 else np.nan
            camarilla_h3[i] = camarilla_h3[i-1] if i > 0 else np.nan
            camarilla_l3[i] = camarilla_l3[i-1] if i > 0 else np.nan
            camarilla_h2[i] = camarilla_h2[i-1] if i > 0 else np.nan
            camarilla_l2[i] = camarilla_l2[i-1] if i > 0 else np.nan
            camarilla_h1[i] = camarilla_h1[i-1] if i > 0 else np.nan
            camarilla_l1[i] = camarilla_l1[i-1] if i > 0 else np.nan
            camarilla_close[i] = camarilla_close[i-1] if i > 0 else np.nan
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros(len(df_1d['volume'].values))
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_20[20:]
    vol_ratio_1d[:20] = 1.0
    
    # Align volume ratio to 6h
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate weekly EMA20 on 1d data (using Friday's close as weekly proxy)
    # Simplified: EMA20 of 1d close
    weekly_ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema_20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    bars_since_entry = 0
    
    warmup = 100  # Warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_aligned[i] > 2.0
        
        # --- Weekly Trend Filter ---
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # --- Camarilla Levels (using previous bar's levels to avoid look-ahead) ---
        # Use levels from previous bar (already calculated from previous 1d bar)
        h4 = camarilla_h4[i-1] if i > 0 else np.nan
        l4 = camarilla_l4[i-1] if i > 0 else np.nan
        h3 = camarilla_h3[i-1] if i > 0 else np.nan
        l3 = camarilla_l3[i-1] if i > 0 else np.nan
        h2 = camarilla_h2[i-1] if i > 0 else np.nan
        l2 = camarilla_l2[i-1] if i > 0 else np.nan
        h1 = camarilla_h1[i-1] if i > 0 else np.nan
        l1 = camarilla_l1[i-1] if i > 0 else np.nan
        pivot = camarilla_close[i-1] if i > 0 else np.nan
        
        if np.isnan(h4) or np.isnan(l4) or np.isnan(h3) or np.isnan(l3):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions based on entry type
            if position_side > 0:  # Long
                # Mean reversion long: exit at H3 (take profit)
                if not weekly_uptrend and close[i] >= h3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Breakout long: exit if price revisits pivot (failed breakout)
                elif weekly_uptrend and close[i] <= pivot:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                # Mean reversion short: exit at L3 (take profit)
                if not weekly_downtrend and close[i] <= l3:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Breakout short: exit if price revisits pivot (failed breakout)
                elif weekly_downtrend and close[i] >= pivot:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Mean Reversion Long: Price touches S3/S4 + weekly downtrend (bear market bounce) + volume spike
        mean_rev_long = ((close[i] <= l3 or close[i] <= l4) and 
                         weekly_downtrend and volume_spike)
        
        # Mean Reversion Short: Price touches R3/R4 + weekly uptrend (bull market pullback) + volume spike
        mean_rev_short = ((close[i] >= h3 or close[i] >= h4) and 
                          weekly_uptrend and volume_spike)
        
        # Breakout Long: Price breaks above H4 + weekly uptrend + volume spike
        breakout_long = (close[i] > h4 and weekly_uptrend and volume_spike)
        
        # Breakout Short: Price breaks below L4 + weekly downtrend + volume spike
        breakout_short = (close[i] < l4 and weekly_downtrend and volume_spike)
        
        if mean_rev_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            bars_since_entry = 0
            signals[i] = SIZE
        elif mean_rev_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals