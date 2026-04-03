#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h capture strong momentum. Confirmed by:
1. 1d HMA(21) trend filter (price above/below HMA for trend alignment)
2. 1d volume spike (>1.8x 20-period average) to ensure participation
3. ATR(14) stoploss (2.5x) for risk management
This structure works in bull/bear markets by trading breakouts in HTF trend direction.
Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) and avoids overtrading via strict volume/HMA filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Donchian Channel (20-period) - using previous 20 bars only (no look-ahead)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[20:] = high_series.rolling(window=20, min_periods=20).max().shift(1).values[20:]
        donchian_low[20:] = low_series.rolling(window=20, min_periods=20).min().shift(1).values[20:]
        # For first 20 bars, use expanding window (no look-ahead)
        for i in range(20):
            donchian_high[i] = high_series.iloc[:i+1].max()
            donchian_low[i] = low_series.iloc[:i+1].min()
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital) to reduce drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1d HMA21 ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume confirmation and uptrend (price > HMA)
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Price breaks below Donchian low with volume confirmation and downtrend (price < HMA)
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_hma
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h capture strong momentum. Confirmed by:
1. 1d HMA(21) trend filter (price above/below HMA for trend alignment)
2. 1d volume spike (>1.8x 20-period average) to ensure participation
3. ATR(14) stoploss (2.5x) for risk management
This structure works in bull/bear markets by trading breakouts in HTF trend direction.
Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag.
Uses discrete position sizing (0.25) and avoids overtrading via strict volume/HMA filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Donchian Channel (20-period) - using previous 20 bars only (no look-ahead)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[20:] = high_series.rolling(window=20, min_periods=20).max().shift(1).values[20:]
        donchian_low[20:] = low_series.rolling(window=20, min_periods=20).min().shift(1).values[20:]
        # For first 20 bars, use expanding window (no look-ahead)
        for i in range(20):
            donchian_high[i] = high_series.iloc[:i+1].max()
            donchian_low[i] = low_series.iloc[:i+1].min()
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital) to reduce drawdown
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1d HMA21 ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume confirmation and uptrend (price > HMA)
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Price breaks below Donchian low with volume confirmation and downtrend (price < HMA)
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_hma
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals