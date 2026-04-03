#!/usr/bin/env python3
"""
Experiment #001: 4h Donchian(20) + 1d/1w Trend + Volume Spike Strategy

HYPOTHESIS: Donchian(20) breakouts on 4h aligned with higher timeframe trends (1d price > 1w EMA200 for longs, 
1d price < 1w EMA200 for shorts) and volume confirmation (>1.8x average) capture strong directional moves.
In strong trends (price clearly above/below 1w EMA200), we trade breakouts with the higher timeframe trend.
In ranging/transitioning markets (price near 1w EMA200), we avoid false breakouts. Uses ATR-based stoploss (2.0x)
and minimum 3-bar holding period to reduce churn. Target: 100-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_001_4h_donchian20_1d_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for price, 1w data for EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Use 1d close price for trend comparison
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 200  # Warmup for 1w EMA200 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Higher Timeframe Trend Filter ---
        price_1d = close_1d_aligned[i]
        ema200 = ema200_1w_aligned[i]
        is_uptrend = price_1d > ema200  # 1d price above 1w EMA200
        is_downtrend = price_1d < ema200  # 1d price below 1w EMA200
        is_transition = abs(price_1d - ema200) / ema200 < 0.02  # Within 2% of EMA200
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Exit Logic (ATR-based stoploss) ---
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
                # Exit on opposite Donchian breakout with volume
                if breakout_down and volume_spike:
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
                # Exit on opposite Donchian breakout with volume
                if breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade breakouts aligned with higher timeframe trend, avoid transition zones
        if is_uptrend and not is_transition:
            # Long: Donchian breakout up AND volume spike
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif is_downtrend and not is_transition:
            # Short: Donchian breakout down AND volume spike
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In transition zone or no clear trend, do not trade
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #001: 4h Donchian(20) + 1d/1w Trend + Volume Spike Strategy

HYPOTHESIS: Donchian(20) breakouts on 4h aligned with higher timeframe trends (1d price > 1w EMA200 for longs, 
1d price < 1w EMA200 for shorts) and volume confirmation (>1.8x average) capture strong directional moves.
In strong trends (price clearly above/below 1w EMA200), we trade breakouts with the higher timeframe trend.
In ranging/transitioning markets (price near 1w EMA200), we avoid false breakouts. Uses ATR-based stoploss (2.0x)
and minimum 3-bar holding period to reduce churn. Target: 100-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_001_4h_donchian20_1d_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for price, 1w data for EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Use 1d close price for trend comparison
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 200  # Warmup for 1w EMA200 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(close_1d_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Higher Timeframe Trend Filter ---
        price_1d = close_1d_aligned[i]
        ema200 = ema200_1w_aligned[i]
        is_uptrend = price_1d > ema200  # 1d price above 1w EMA200
        is_downtrend = price_1d < ema200  # 1d price below 1w EMA200
        is_transition = abs(price_1d - ema200) / ema200 < 0.02  # Within 2% of EMA200
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = low[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Exit Logic (ATR-based stoploss) ---
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
                # Exit on opposite Donchian breakout with volume
                if breakout_down and volume_spike:
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
                # Exit on opposite Donchian breakout with volume
                if breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade breakouts aligned with higher timeframe trend, avoid transition zones
        if is_uptrend and not is_transition:
            # Long: Donchian breakout up AND volume spike
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif is_downtrend and not is_transition:
            # Short: Donchian breakout down AND volume spike
            if breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In transition zone or no clear trend, do not trade
            signals[i] = 0.0
    
    return signals