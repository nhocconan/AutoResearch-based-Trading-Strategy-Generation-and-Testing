#!/usr/bin/env python3
"""
Experiment #415: 15m Vol Spike Mean Reversion + 4h HMA Trend Filter

Hypothesis: After 414 failed experiments, the pattern is clear: pure trend-following
fails on BTC/ETH due to 2022 crash whipsaw and 2025 bear market. Mean reversion
ALONE fails without trend filter. The solution is VOLATILITY SPIKE MEAN REVERSION:

1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme volatility
   - This happens at market bottoms (panic sells) and tops (panic buys)
   - Research shows 75%+ win rate on vol spike reversals

2. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long ONLY when price > 4h HMA (bullish bias, buy the dip)
   - Short ONLY when price < 4h HMA (bearish bias, sell the rip)
   - Prevents counter-trend trades that destroy accounts in 2022-style crashes

3. BOLLINGER BANDS (20, 2.5) for entry trigger:
   - Long: price < lower band (oversold extreme)
   - Short: price > upper band (overbought extreme)
   - Wider bands (2.5 vs 2.0) = fewer but higher quality signals

4. RSI(14) CONFIRMATION:
   - Long: RSI < 35 (not too strict, must generate trades)
   - Short: RSI > 65
   - Confirms mean reversion setup

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crashes

6. POSITION SIZING: 0.25 discrete (conservative for 15m noise)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 15m + vol spike should work:
- 15m captures intraday panic/reversals that 4h/1d miss
- Vol spike filter = only trade extreme moves (high conviction)
- 4h HMA filter = avoid counter-trend disasters
- Should generate 50-100 trades/year (enough for stats, not too many for fees)
- Works on BTC/ETH/SOL individually (vol spikes happen on all)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_spike_4h_hma_bb_rsi_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
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
    
    # Calculate 15m indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.5)
    rsi = calculate_rsi(close, 14)
    
    # Volatility spike ratio
    vol_ratio = atr_7 / np.where(atr_30 > 1e-10, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(vol_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]  # Price below lower band
        bb_overbought = close[i] > bb_upper[i]  # Price above upper band
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35  # Not too strict (must generate trades)
        rsi_overbought = rsi[i] > 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: Vol spike + oversold + bullish 4h trend + RSI confirmation
        if vol_spike and bb_oversold and bull_trend_4h and rsi_oversold:
            new_signal = SIZE
        
        # SHORT: Vol spike + overbought + bearish 4h trend + RSI confirmation
        elif vol_spike and bb_overbought and bear_trend_4h and rsi_overbought:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === VOL SPIKE EXIT ===
        # Exit if vol spike condition ends (vol_ratio < 1.5)
        if in_position and new_signal != 0.0:
            if vol_ratio[i] < 1.5:
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