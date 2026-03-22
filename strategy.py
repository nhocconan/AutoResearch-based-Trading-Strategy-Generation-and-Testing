#!/usr/bin/env python3
"""
Experiment #547: 15m Volume-Confirmed RSI with 4h HMA Trend Bias

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. 15m timeframe needs STRONG filters to avoid noise (most 15m strategies failed)
2. Volume confirmation (taker_buy_volume ratio) filters false breakouts - UNIQUE EDGE
3. 4h HMA trend bias prevents counter-trend entries (proven in best strategies)
4. RSI(7) extremes with volume confirmation = high probability mean reversion
5. Asymmetric sizing: larger positions when 4h trend confirms, smaller when against
6. Tight 2*ATR stoploss for 15m (faster timeframe needs tighter stops)

Why this should work on 15m:
- Volume ratio shows institutional flow direction (unique data column available)
- 4h HMA is available via mtf_data helper with proper alignment
- RSI(7) is faster than RSI(14), catches 15m swings better
- Asymmetric sizing reduces risk when trading against HTF trend
- Fewer but higher quality trades = less fee drag

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_volume_rsi_4h_hma_asymmetric_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (institutional flow indicator)."""
    ratio = taker_buy_volume / np.maximum(volume, 1)
    return ratio

def calculate_zscore(values, period=20):
    """Calculate rolling z-score for normalization."""
    values_s = pd.Series(values)
    rolling_mean = values_s.rolling(window=period, min_periods=period).mean()
    rolling_std = values_s.rolling(window=period, min_periods=period).std()
    zscore = (values_s - rolling_mean) / np.maximum(rolling_std, 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    vol_ratio_zscore = calculate_zscore(vol_ratio, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_CONFIRMED = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio_zscore[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # High taker buy ratio = bullish institutional flow
        # Low taker buy ratio = bearish institutional flow
        vol_bullish = vol_ratio[i] > 0.55  # More than 55% taker buys
        vol_bearish = vol_ratio[i] < 0.45  # Less than 45% taker buys
        
        # Volume z-score confirmation (extreme volume = stronger signal)
        vol_extreme = np.abs(vol_ratio_zscore[i]) > 1.0
        
        # === RSI EXTREMES (faster period for 15m) ===
        rsi_oversold = rsi_7[i] < 25  # Faster than RSI(14)<30
        rsi_overbought = rsi_7[i] > 75  # Faster than RSI(14)>70
        
        # === ENTRY LOGIC WITH ASYMMETRIC SIZING ===
        new_signal = 0.0
        
        # Long: RSI oversold + volume bullish + 4h trend confirmation
        if rsi_oversold and vol_bullish:
            if bull_bias:
                # 4h trend confirms = larger position
                new_signal = SIZE_CONFIRMED
            else:
                # Trading against 4h trend = smaller position (mean reversion only)
                new_signal = SIZE_BASE
        
        # Short: RSI overbought + volume bearish + 4h trend confirmation
        elif rsi_overbought and vol_bearish:
            if bear_bias:
                # 4h trend confirms = larger position
                new_signal = -SIZE_CONFIRMED
            else:
                # Trading against 4h trend = smaller position (mean reversion only)
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips strongly against position
        if in_position and new_signal != 0.0:
            hma_distance = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
            if position_side > 0 and hma_distance < -0.02:  # Price 2% below 4h HMA
                new_signal = 0.0
            if position_side < 0 and hma_distance > 0.02:  # Price 2% above 4h HMA
                new_signal = 0.0
        
        # === RSI NORMALIZATION EXIT ===
        # Exit when RSI returns to neutral (mean reversion complete)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi_7[i] > 55:  # Long exit when RSI recovers
                new_signal = 0.0
            if position_side < 0 and rsi_7[i] < 45:  # Short exit when RSI recovers
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals