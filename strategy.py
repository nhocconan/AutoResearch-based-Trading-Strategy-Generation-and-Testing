#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d/1w HTF — Dual Regime (Trend + Mean Revert)

Hypothesis: Higher timeframes (12h) reduce noise and fee drag. Using 1d/1w for 
trend direction + 12h for entries should capture major moves while avoiding 
whipsaws. Dual regime adapts to market conditions.

Key components:
1. 1d HMA for primary trend direction (bullish/bearish)
2. 1w HMA for major regime filter (only trade with weekly trend)
3. 12h ADX to detect trending vs ranging (ADX>25=trend, ADX<20=range)
4. Trend regime: Enter on 12h HMA pullback to 12h EMA21 in direction of 1d trend
5. Range regime: Mean revert at BB extremes with RSI confirmation
6. ATR trailing stop (2.5x) for risk management

Why this might work:
- 12h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Dual regime adapts to both trending 2021 and ranging 2022-2024
- 1w filter prevents fighting major trend (critical for 2025 bear market)
- Position size 0.28 (conservative, allows surviving 77% crash with -27% DD)

Entry conditions (LOOSE to ensure trades):
- Trend long: 1d HMA bullish + 1w HMA bullish + 12h price>EMA21 + RSI<60
- Trend short: 1d HMA bearish + 1w HMA bearish + 12h price<EMA21 + RSI>40
- Range long: ADX<20 + BB_pct_b<0.15 + RSI<35
- Range short: ADX<20 + BB_pct_b>0.85 + RSI>65

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_1d_1w_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    pct_b = (close - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper.values, lower.values, pct_b.values, bandwidth.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for major regime
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    
    # 12h HMA for entry timing
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(rsi_14[i]) or np.isnan(ema_21[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND DIRECTION ===
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # 1d HMA slope
        hma_1d_slope_up = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_down = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # === 1W MAJOR REGIME ===
        hma_1w_bullish = close[i] > hma_1w_aligned[i]
        hma_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (ADX) ===
        trending_regime = adx_14[i] > 25
        ranging_regime = adx_14[i] < 20
        
        # === 12H ENTRY TIMING ===
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        price_above_hma12h = close[i] > hma_12h[i]
        price_below_hma12h = close[i] < hma_12h[i]
        
        # === TREND REGIME ENTRIES ===
        trend_long = False
        trend_short = False
        
        if trending_regime:
            # Long: 1d bullish + 1w bullish + pullback to EMA21 + RSI not overbought
            trend_long = (hma_1d_bullish or hma_1d_slope_up) and \
                         (hma_1w_bullish or hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else True) and \
                         price_above_ema21 and \
                         rsi_14[i] < 65 and \
                         rsi_14[i] > 35
            
            # Short: 1d bearish + 1w bearish + rally to EMA21 + RSI not oversold
            trend_short = (hma_1d_bearish or hma_1d_slope_down) and \
                          (hma_1w_bearish or hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else True) and \
                          price_below_ema21 and \
                          rsi_14[i] > 35 and \
                          rsi_14[i] < 65
        
        # === RANGE REGIME ENTRIES ===
        range_long = False
        range_short = False
        
        if ranging_regime:
            # Long: BB lower band + RSI oversold
            range_long = bb_pct_b[i] < 0.15 and rsi_14[i] < 35
            
            # Short: BB upper band + RSI overbought
            range_short = bb_pct_b[i] > 0.85 and rsi_14[i] > 65
        
        # === COMBINE SIGNALS ===
        new_signal = 0.0
        
        if trend_long or range_long:
            new_signal = POSITION_SIZE
        
        if trend_short or range_short:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME/TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 1d turns bearish or ADX spikes (trend breaking)
            if hma_1d_bearish and adx_14[i] > 30:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d turns bullish or ADX spikes
            if hma_1d_bullish and adx_14[i] > 30:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals