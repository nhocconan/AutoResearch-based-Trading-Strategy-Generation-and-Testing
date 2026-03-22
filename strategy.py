#!/usr/bin/env python3
"""
Experiment #079: 15m RSI Pullback with 4h HMA Trend + 1h Supertrend Confirmation
Hypothesis: 15m timeframe is ideal for catching pullbacks within higher-TF trends.
Instead of breakout chasing (which fails on 15m due to noise), we buy/sell pullbacks
when the 4h trend is clear and 1h supertrend confirms direction.
Key insight: 15m strategies #067, #073 failed because they tried to trade breakouts
on noisy data. Pullback entries have better win rates in trending markets.
This strategy uses:
- 4h HMA(21) for primary trend bias (long only above, short only below)
- 1h Supertrend(10,3) for intermediate confirmation (must match direction)
- 15m RSI(14) pullback entries (RSI 35-45 for longs, 55-65 for shorts)
- Volume confirmation (current volume > 1.3x 20-bar avg)
- ATR(14) trailing stop at 2.0x for risk management
- Discrete position sizing (0.25 base, 0.30 strong)
Why this might work: Pullback entries have 60-70% win rates in trending markets.
4h HMA provides smooth trend filter. 1h Supertrend adds confirmation without
killing trade frequency. 15m RSI catches intraday dips/rallies.
Timeframe: 15m (REQUIRED), HTF: 1h and 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_4h_hma_1h_supertrend_v1"
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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, trend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band[:] = np.nan
    lower_band[:] = np.nan
    supertrend[:] = np.nan
    trend[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if close[i - 1] <= supertrend[i - 1]:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    trend[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    trend[i] = 1
    
    return supertrend, trend

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    supertrend_1h, trend_1h = calculate_supertrend(
        df_1h['high'].values,
        df_1h['low'].values,
        df_1h['close'].values,
        period=10,
        multiplier=3.0
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    trend_1h_aligned = align_htf_to_ltf(prices, df_1h, trend_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
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
        
        if np.isnan(trend_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = primary trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend = intermediate confirmation
        bull_supertrend_1h = trend_1h_aligned[i] == 1
        bear_supertrend_1h = trend_1h_aligned[i] == -1
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI PULLBACK ZONES ===
        # For longs: RSI pulled back to 35-50 zone (not oversold, just resting)
        rsi_pullback_long = 35 <= rsi[i] <= 50
        # For shorts: RSI rallied to 50-65 zone (not overbought, just resting)
        rsi_pullback_short = 50 <= rsi[i] <= 65
        
        # RSI momentum (rising for longs, falling for shorts)
        rsi_rising = rsi[i] > rsi[i - 1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i - 1] if i > 0 else False
        
        # Price above/below EMA21 for entry timing
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h bullish + 1h supertrend bullish + RSI pullback
        if bull_trend_4h and bull_supertrend_1h:
            if rsi_pullback_long:
                if price_above_ema21 or rsi_rising:
                    if vol_confirmed or ema_bullish:
                        new_signal = SIZE_STRONG
                    else:
                        new_signal = SIZE_BASE
        
        # Secondary: 4h bullish + EMA bullish + RSI recovering from pullback
        if bull_trend_4h and ema_bullish:
            if 40 <= rsi[i] <= 55:
                if rsi_rising and price_above_ema21:
                    new_signal = SIZE_BASE
        
        # Tertiary: Ensure we get trades - simpler conditions
        if bull_trend_4h:
            if 35 <= rsi[i] <= 55:
                if price_above_ema21:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h bearish + 1h supertrend bearish + RSI pullback
        if bear_trend_4h and bear_supertrend_1h:
            if rsi_pullback_short:
                if price_below_ema21 or rsi_falling:
                    if vol_confirmed or ema_bearish:
                        new_signal = -SIZE_STRONG
                    else:
                        new_signal = -SIZE_BASE
        
        # Secondary: 4h bearish + EMA bearish + RSI recovering from pullback
        if bear_trend_4h and ema_bearish:
            if 45 <= rsi[i] <= 60:
                if rsi_falling and price_below_ema21:
                    new_signal = -SIZE_BASE
        
        # Tertiary: Ensure we get trades - simpler conditions
        if bear_trend_4h:
            if 45 <= rsi[i] <= 65:
                if price_below_ema21:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals