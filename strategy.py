#!/usr/bin/env python3
"""
Experiment #347: 6h Volume-Weighted RSI + 1d Trend Filter

HYPOTHESIS: Volume-weighted RSI (VW-RSI) improves upon standard RSI by weighting price changes with volume,
making it more responsive to institutional participation. Combined with 1d EMA200 trend filter and
ATR-based volatility filter, this strategy captures momentum bursts in both bull and bear markets.
The 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
VW-RSI < 30 signals oversold conditions for longs; VW-RSI > 70 signals overbought for shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_347_6h_vwrsi_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicators: Volume-Weighted RSI(14) ===
    # Calculate price changes
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gains = gains * volume
    vol_losses = losses * volume
    
    # Calculate smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gains).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_vol_loss = pd.Series(vol_losses).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 100)
    vwrsi = 100 - (100 / (1 + rs))
    
    # === 6h Indicators: ATR(14) for stoploss and volatility filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (0.5 * atr_ma_50)  # Only trade when volatility is above 50% of MA
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 200  # Warmup for 1d EMA200 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vwrsi[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # --- Price and Levels ---
        price = close[i]
        rsi = vwrsi[i]
        ema200 = ema200_1d_aligned[i]
        atr = atr_14[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions: VW-RSI mean reversion
            if position_side > 0 and rsi > 60:  # Exit long on overbought
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and rsi < 40:  # Exit short on oversold
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: VW-RSI < 30 (oversold) + price > 1d EMA200 (uptrend) + volatility filter
        long_entry = (rsi < 30) and (price > ema200) and vol_filter[i]
        
        # Short: VW-RSI > 70 (overbought) + price < 1d EMA200 (downtrend) + volatility filter
        short_entry = (rsi > 70) and (price < ema200) and vol_filter[i]
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #347: 6h Volume-Weighted RSI + 1d Trend Filter

HYPOTHESIS: Volume-weighted RSI (VW-RSI) improves upon standard RSI by weighting price changes with volume,
making it more responsive to institutional participation. Combined with 1d EMA200 trend filter and
ATR-based volatility filter, this strategy captures momentum bursts in both bull and bear markets.
The 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
VW-RSI < 30 signals oversold conditions for longs; VW-RSI > 70 signals overbought for shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_347_6h_vwrsi_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicators: Volume-Weighted RSI(14) ===
    # Calculate price changes
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gains = gains * volume
    vol_losses = losses * volume
    
    # Calculate smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gains).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_vol_loss = pd.Series(vol_losses).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 100)
    vwrsi = 100 - (100 / (1 + rs))
    
    # === 6h Indicators: ATR(14) for stoploss and volatility filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (0.5 * atr_ma_50)  # Only trade when volatility is above 50% of MA
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 200  # Warmup for 1d EMA200 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vwrsi[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # --- Price and Levels ---
        price = close[i]
        rsi = vwrsi[i]
        ema200 = ema200_1d_aligned[i]
        atr = atr_14[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions: VW-RSI mean reversion
            if position_side > 0 and rsi > 60:  # Exit long on overbought
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and rsi < 40:  # Exit short on oversold
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: VW-RSI < 30 (oversold) + price > 1d EMA200 (uptrend) + volatility filter
        long_entry = (rsi < 30) and (price > ema200) and vol_filter[i]
        
        # Short: VW-RSI > 70 (overbought) + price < 1d EMA200 (downtrend) + volatility filter
        short_entry = (rsi > 70) and (price < ema200) and vol_filter[i]
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals