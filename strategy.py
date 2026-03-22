#!/usr/bin/env python3
"""
Experiment #213: 1h MACD Momentum + 4h HMA Trend + BB Squeeze + Volume Filter

Hypothesis: 1h timeframe needs STRONGER filtering than 4h due to more noise.
This strategy combines:
1. 4h HMA(21) as primary trend filter (HTF bias - call ONCE before loop)
2. 1h MACD histogram momentum for entry timing (faster signal)
3. Bollinger Band Width squeeze detection (low vol = breakout potential)
4. Volume confirmation (taker_buy_volume ratio > 0.55 for longs)
5. ATR(14) trailing stop at 2.5x for risk management

Why this might work on 1h:
- 4h HMA provides stable trend bias (proven in current best strategy)
- MACD histogram captures momentum shifts faster than EMA crossover
- BB squeeze identifies consolidation before breakouts (reduces whipsaws)
- Volume filter confirms genuine moves vs fake breakouts
- Conservative sizing (0.25) controls 2022-style crash drawdown

Learning from failures:
- #201 (1h KAMA): Sharpe=-1.202 - KAMA alone too slow for 1h noise
- #207 (1h RSI mean-rev): Sharpe=-9.084 - mean-rev fails on 1h crypto
- #211 (15m MACD): Sharpe=-2.490 - too fast, needs stronger HTF filter
- Current best uses 4h+1d, this uses 1h+4h (faster but filtered)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (max 0.30)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_macd_4h_hma_bb_squeeze_vol_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values, std.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (normalized)."""
    width = (upper - lower) / sma
    return width

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio."""
    ratio = taker_buy_volume / (volume + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_sma, bb_std = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Calculate BB Width percentile for squeeze detection (rolling 100 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
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
        
        if np.isnan(macd_hist[i]) or np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (STRONG filter for 1h)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === BB SQUEEZE DETECTION ===
        # BB Width percentile < 0.3 = low volatility (squeeze = breakout potential)
        bb_squeeze = bb_width_percentile[i] < 0.3
        
        # === MACD MOMENTUM ===
        # MACD histogram crossing above 0 = bullish momentum
        # MACD histogram crossing below 0 = bearish momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0  # Fresh cross
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0  # Fresh cross
        
        # MACD histogram positive/negative (sustained momentum)
        macd_hist_positive = macd_hist[i] > 0
        macd_hist_negative = macd_hist[i] < 0
        
        # === EMA CONFIRMATION ===
        # EMA21 > EMA50 = bullish trend structure
        # EMA21 < EMA50 = bearish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        # Taker buy ratio > 0.55 = buying pressure
        # Taker buy ratio < 0.45 = selling pressure
        vol_bullish = vol_ratio[i] > 0.55
        vol_bearish = vol_ratio[i] < 0.45
        
        # === PRICE POSITION IN BB ===
        # Price near upper band = bullish
        # Price near lower band = bearish
        price_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        price_upper = price_position > 0.6
        price_lower = price_position < 0.4
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + (BB squeeze OR MACD fresh cross) + volume/price confirmation
        # More flexible to ensure enough trades on 1h
        if bull_trend_4h:
            # Primary entry: BB squeeze + MACD momentum + volume
            if bb_squeeze and macd_hist_positive:
                if vol_bullish or price_upper or ema_bullish:
                    new_signal = SIZE_BASE
            
            # Secondary entry: MACD fresh cross (momentum shift)
            if macd_bullish:
                if vol_bullish or ema_bullish:
                    new_signal = SIZE_BASE
        
        # Short: 4h bearish + (BB squeeze OR MACD fresh cross) + volume/price confirmation
        if bear_trend_4h:
            # Primary entry: BB squeeze + MACD momentum + volume
            if bb_squeeze and macd_hist_negative:
                if vol_bearish or price_lower or ema_bearish:
                    new_signal = -SIZE_BASE
            
            # Secondary entry: MACD fresh cross (momentum shift)
            if macd_bearish:
                if vol_bearish or ema_bearish:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals