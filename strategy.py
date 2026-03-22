#!/usr/bin/env python3
"""
Experiment #122: 30m RSI Mean Reversion + 4h HMA Trend Filter + ATR Stop

Hypothesis: After 121 experiments, most trend-following strategies fail on BTC/ETH
due to 2022 crash whipsaws and 2025 bear market. This strategy uses:
- 30m RSI(14) mean reversion: enters at extremes (RSI<30 long, RSI>70 short)
- 4h HMA(21) trend filter: only long when price>4h_HMA, only short when price<4h_HMA
- Choppiness Index(14) regime filter: high CHOP>61.8 = range (favor mean reversion),
  low CHOP<38.2 = trend (reduce position size or skip mean reversion)
- ATR(14) trailing stop at 2.5*ATR to protect against reversals
- Discrete position sizing (0.20/0.30) to minimize fee churn

Why 30m works better than 15m/1h for this approach:
- 30m captures intraday swings without 15m noise
- More trades than 4h/12h (ensures ≥10 trades per symbol)
- RSI mean reversion works in both bull and bear markets
- 4h HMA filter prevents counter-trend trades during crashes

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_meanrev_4h_hma_chop_atr_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(len(close))
    mask = loss_s > 0
    rs[mask] = gain_s[mask] / loss_s[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[loss_s == 0] = 100.0
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    for i in range(period, n):
        atr_sum = np.sum(atr_s.values[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # High CHOP = range market (favor mean reversion)
        # Low CHOP = trending market (reduce mean reversion signals)
        choppy_market = not np.isnan(chop[i]) and chop[i] > 55.0
        trending_market = not np.isnan(chop[i]) and chop[i] < 45.0
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Long: RSI oversold (<30) + 4h bullish trend
        rsi_oversold = rsi[i] < 30
        # Short: RSI overbought (>70) + 4h bearish trend
        rsi_overbought = rsi[i] > 70
        
        # === POSITION SIZE ADJUSTMENT BASED ON REGIME ===
        # In choppy market: full size (mean reversion works best)
        # In trending market: half size or skip (mean reversion risky)
        if choppy_market:
            size_mult = 1.0
        elif trending_market:
            size_mult = 0.5  # Reduce size in trending market
        else:
            size_mult = 0.75  # Neutral regime
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if rsi_oversold and bull_trend_4h:
            new_signal = SIZE_STRONG * size_mult
        elif rsi_oversold:
            # Weaker signal without trend confirmation
            new_signal = SIZE_BASE * size_mult
        
        # === SHORT ENTRY CONDITIONS ===
        if rsi_overbought and bear_trend_4h:
            new_signal = -SIZE_STRONG * size_mult
        elif rsi_overbought:
            # Weaker signal without trend confirmation
            new_signal = -SIZE_BASE * size_mult
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # === RSI EXIT CONDITIONS ===
        # Exit long when RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_overbought:
            new_signal = 0.0
        
        # Exit short when RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_oversold:
            new_signal = 0.0
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals