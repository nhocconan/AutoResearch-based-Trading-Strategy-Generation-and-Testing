#!/usr/bin/env python3
"""
Experiment #200: 30m KAMA Trend + 4h HMA Filter + Volume Confirmation + ATR Stop

Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides stable
trend bias. KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends,
slow in ranges. Volume confirmation filters false breakouts. ATR trailing stop
protects capital. This avoids the mean-reversion trap that failed in #194.

Why 30m might work:
- 30m = 48 bars/day, captures swing moves without 15m noise
- KAMA adapts to market regime automatically (no ADX threshold needed)
- 4h HMA filter prevents counter-trend trades (proven in best strategy)
- Volume spike (1.5x avg) confirms genuine moves, filters fakeouts
- Flexible RSI filter (>45/<55) ensures enough trades vs strict extremes

Learning from failures:
- #188 (30m EMA+Chop): Sharpe=-1.132 - chopiness index unreliable
- #194 (30m RSI BB meanrev): Sharpe=-2.588 - mean reversion fails on crypto
- #193/#199 (15m pullback): Sharpe=-3.5 to -4.8 - pullbacks get destroyed
- Trend-following with HTF filter works, mean-reversion doesn't
- Need flexible entries to ensure ≥10 trades/symbol

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels (balances trade count vs DD)
Stoploss: 2.2 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_volume_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - fast in trends, slow in ranges.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast - slow) + slow]^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close_s - close_s.shift(er_period))
    sum_changes = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er.values * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize at price
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period]
    
    return kama

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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / (vol_ma.values + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=10, fast_period=5, slow_period=50)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    
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
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (proven effective)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA CROSSOVER ===
        # Fast KAMA > Slow KAMA = bullish momentum
        # Fast KAMA < Slow KAMA = bearish momentum
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.3x average confirms genuine move
        volume_confirmed = vol_ratio[i] > 1.3
        
        # === RSI FILTER (flexible, not extreme) ===
        # RSI > 45 = not oversold (for longs)
        # RSI < 55 = not overbought (for shorts)
        rsi_ok_long = rsi[i] > 45
        rsi_ok_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS (flexible to ensure trades) ===
        # Long: 4h bullish + KAMA bullish + (volume OR RSI ok)
        if bull_trend_4h and kama_bullish:
            if volume_confirmed or rsi_ok_long:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + KAMA bearish + (volume OR RSI ok)
        if bear_trend_4h and kama_bearish:
            if volume_confirmed or rsi_ok_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.2 * ATR trailing ===
        # Check stoploss on EXISTING position
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.2 * ATR below highest close
                stoploss_price = highest_close - 2.2 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss hit - override entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.2 * ATR above lowest close
                stoploss_price = lowest_close + 2.2 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss hit - override entry signal
        
        # === UPDATE POSITION TRACKING ===
        # Track position state based on signal (for next bar's stoploss calc)
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Signal is 0 - check if we should exit or maintain stoploss tracking
            if in_position:
                # Position still active for stoploss tracking (signal will be 0 next bar if stop hit)
                if position_side > 0 and close[i] > highest_close:
                    highest_close = close[i]
                elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                    lowest_close = close[i]
                # If stoploss was hit, in_position will be set to False next iteration
                # when new_signal stays 0 and we detect exit
            else:
                # No position, reset tracking
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        # Detect position exit for tracking reset
        if in_position and new_signal == 0.0:
            # Check if this is a genuine exit (stoploss or signal reversal to flat)
            # We need to track if position was active last bar
            pass  # Keep tracking for stoploss
        
        # Simple position exit detection: if signal goes from non-zero to zero
        if i > 100 and signals[i-1] != 0.0 and new_signal == 0.0:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals