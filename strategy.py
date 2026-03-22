#!/usr/bin/env python3
"""
Experiment #327: 1h KAMA Trend with 4h HMA Bias and RSI Pullback

Hypothesis: After analyzing failures #315-#326, mean reversion (CRSI, Fisher) 
consistently fails on sub-4h timeframes. The only profitable strategy (#316) 
used 4h with regime + HTF HMA. For 1h, I'll use:

1. 4h HMA(21) for directional bias (proven edge from #316)
2. 1h KAMA(14) for adaptive trend entry (adjusts to volatility/efficiency)
3. RSI(14) for pullback confirmation (not extreme values, just >50/<50)
4. ATR(14) trailing stoploss at 2.5x (proven from successful strategies)
5. LOOSE entry conditions to ensure >=10 trades (learned from #311)

Why KAMA instead of EMA? KAMA adapts to market noise - faster in trends, 
slower in chop. This should reduce whipsaws in 1h timeframe while maintaining
responsiveness. Combined with 4h HMA bias, this creates a robust MTF system.

Position sizing: 0.25 base, 0.30 when 4h trend strong
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_pullback_atr_v1"
timeframe = "1h"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency ratio (ER).
    ER = |change| / sum(|changes|) over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize with price
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND ===
        # Price above KAMA = bullish, below = bearish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (trend strength)
        kama_slope_bull = kama[i] > kama[i-1] if i > 0 else False
        kama_slope_bear = kama[i] < kama[i-1] if i > 0 else False
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum, < 50 = bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # RSI pullback zones (not extremes, just confirmation)
        rsi_pullback_long = rsi[i] > 45  # Allow entry even on slight pullback
        rsi_pullback_short = rsi[i] < 55
        
        # === DETERMINE POSITION SIZE ===
        # Stronger size when 4h trend aligns with entry direction
        position_size = SIZE_BASE
        if bull_trend_4h and kama_bullish:
            position_size = SIZE_STRONG
        elif bear_trend_4h and kama_bearish:
            position_size = SIZE_STRONG
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades) ===
        new_signal = 0.0
        
        # LONG: 4h bias up + KAMA bullish + RSI confirmation
        # Removed strict crossover requirement - allow trend continuation
        long_conditions = (
            bull_trend_4h and
            kama_bullish and
            rsi_pullback_long
        )
        
        # SHORT: 4h bias down + KAMA bearish + RSI confirmation
        short_conditions = (
            bear_trend_4h and
            kama_bearish and
            rsi_pullback_short
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit when 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === KAMA REVERSAL EXIT ===
        # Exit when KAMA trend flips
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish:
                new_signal = 0.0
            if position_side < 0 and kama_bullish:
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