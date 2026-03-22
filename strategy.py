#!/usr/bin/env python3
"""
Experiment #115: 15m Multi-Timeframe Pullback Strategy with 4h Trend Filter

Hypothesis: After 10+ failed 15m strategies, the key issue is trading against HTF trend.
This strategy uses:
- 4h HMA(21) as STRONG trend filter (only trade in HTF trend direction)
- 1h RSI(14) for pullback timing (enter on weakness in uptrend, strength in downtrend)
- 15m EMA(21) for entry confirmation (price action alignment)
- Volume filter to avoid low-liquidity false signals
- ATR(14) trailing stop at 2.5*ATR for risk management
- Conservative position sizing (0.20-0.30) to survive 2022-style crashes

Why this might work when other 15m strategies failed:
- 4h trend filter prevents counter-trend trades (major cause of 15m failures)
- RSI pullback entries avoid chasing breakouts (reduces whipsaw)
- Volume confirmation filters low-quality signals
- Fewer but higher-quality trades (target 30-50 trades/year)
- Asymmetric logic: easier to stay in trend, harder to reverse

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_4h_hma_volume_atr_v2"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h HTF indicators
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_15m = calculate_ema(close, 21)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_15m[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS (4h HMA) ===
        # STRONG filter: only trade in direction of 4h trend
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to oversold in uptrend
        rsi_oversold_1h = rsi_1h_aligned[i] < 45
        # Short: RSI pulled back to overbought in downtrend
        rsi_overbought_1h = rsi_1h_aligned[i] > 55
        
        # === 15m PRICE ACTION CONFIRMATION ===
        # Price above EMA for long confirmation
        price_above_ema = close[i] > ema_15m[i]
        # Price below EMA for short confirmation
        price_below_ema = close[i] < ema_15m[i]
        
        # === VOLUME FILTER ===
        # Volume must be at least 80% of average (avoid low-liquidity traps)
        volume_ok = volume[i] > vol_sma[i] * 0.8
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # All conditions must align for long entry
        if bull_trend_4h and rsi_oversold_1h and price_above_ema and volume_ok:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + RSI pullback (relax volume for more trades)
        elif bull_trend_4h and rsi_oversold_1h and price_above_ema:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # All conditions must align for short entry
        if bear_trend_4h and rsi_overbought_1h and price_below_ema and volume_ok:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + RSI pullback (relax volume for more trades)
        elif bear_trend_4h and rsi_overbought_1h and price_below_ema:
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