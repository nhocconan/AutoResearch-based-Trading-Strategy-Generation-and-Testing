#!/usr/bin/env python3
"""
Experiment #139: 15m RSI Mean Reversion + 4h HMA Trend Filter + BB Confirmation + ATR Stop

Hypothesis: 15m is too noisy for pure trend following (as seen in exp #133 Sharpe=-5.032).
Instead, use mean reversion on 15m WITH HTF trend filter to avoid counter-trend traps.

Why this might work:
- RSI(14) extremes (<30/>70) on 15m capture short-term oversold/overbought
- 4h HMA(21) provides stable trend bias (only long in bull, short in bear)
- Bollinger Bands (20,2.5) confirm extreme price position
- ATR(14) trailing stop at 2.5*ATR protects from runaway losses
- 15m generates enough trades (>10 per symbol) while HTF filter reduces whipsaw
- Discrete position sizing (0.20/0.30) minimizes fee churn

Key difference from failed 15m strategies:
- Previous 15m strategies tried trend following (breakouts, crossovers)
- This uses MEAN REVERSION with HTF trend filter (proven edge for BTC/ETH)
- Funding rate research shows mean reversion works best in bear/range markets

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_4h_hma_bb_meanrev_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
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

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with wider std_dev for 15m noise."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.5)
    
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
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # RSI < 30 = oversold (potential long)
        # RSI > 70 = overbought (potential short)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Stronger extremes for higher conviction
        rsi_deep_oversold = rsi[i] < 25
        rsi_deep_overbought = rsi[i] > 75
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Price below lower BB = oversold confirmation
        # Price above upper BB = overbought confirmation
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # Price near BB extremes (within 0.5% of band)
        bb_lower_threshold = bb_lower[i] * 1.005
        bb_upper_threshold = bb_upper[i] * 0.995
        price_near_lower_bb = close[i] < bb_lower_threshold
        price_near_upper_bb = close[i] > bb_upper_threshold
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (Mean Reversion in Uptrend) ===
        # Only long when 4h trend is bullish (avoid counter-trend traps)
        if bull_trend_4h:
            # Strong: RSI deep oversold + price below/near lower BB
            if rsi_deep_oversold and (price_below_bb or price_near_lower_bb):
                new_signal = SIZE_STRONG
            # Moderate: RSI oversold + price near lower BB
            elif rsi_oversold and price_near_lower_bb:
                new_signal = SIZE_BASE
            # Weak (ensure trades): RSI oversold in bull trend
            elif rsi_oversold and bull_trend_4h:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (Mean Reversion in Downtrend) ===
        # Only short when 4h trend is bearish (avoid counter-trend traps)
        if bear_trend_4h:
            # Strong: RSI deep overbought + price above/near upper BB
            if rsi_deep_overbought and (price_above_bb or price_near_upper_bb):
                new_signal = -SIZE_STRONG
            # Moderate: RSI overbought + price near upper BB
            elif rsi_overbought and price_near_upper_bb:
                new_signal = -SIZE_BASE
            # Weak (ensure trades): RSI overbought in bear trend
            elif rsi_overbought and bear_trend_4h:
                new_signal = -SIZE_BASE
        
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