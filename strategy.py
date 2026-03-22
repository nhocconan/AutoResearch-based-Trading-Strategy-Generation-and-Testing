#!/usr/bin/env python3
"""
Experiment #460: 1h Primary + 4h/12h HTF — Fisher Transform + KAMA Trend + Volume

Hypothesis: After 459 failed experiments, clear pattern emerges:
1. Fisher Transform catches reversals better than RSI/CRSI in bear/range markets
2. KAMA (Kaufman Adaptive MA) adapts to volatility, less whipsaw than HMA/EMA
3. 4h/12h HTF trend filter provides major direction bias
4. Volume confirmation reduces false breakouts
5. Simpler entry logic = more trades (critical after many 0-trade failures)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform proven edge in research notes for reversal entries
- KAMA efficiency ratio adapts to market regime automatically
- 1h TF with HTF filter = HTF trade frequency with 1h execution precision
- Asymmetric sizing protects in 2022-style crashes
- Target: 40-80 trades/year on 1h (fee drag controlled)

Position sizing: 0.30 long, 0.25 short (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 each symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_htf_vol_asym_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    
    er = signal / (noise + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er.values * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    X = 0.67 * (close - lowest) / (highest - lowest) - 0.33
    
    Long: Fisher crosses above -1.5 (oversold reversal)
    Short: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max().values
    lowest = low_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price position
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    x = 0.67 * (close - lowest) / range_hl - 0.33
    
    # Clamp to avoid log issues
    x = np.clip(x, -0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # KAMA slope (trend direction)
    kama_slope = kama_1h - np.roll(kama_1h, 5)
    kama_slope[:5] = 0
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        if np.isnan(kama_1h[i]) or np.isnan(fisher[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA = bull bias (favor longs)
        # Price below 12h HMA = bear bias (favor shorts)
        bull_htf = close[i] > hma_4h_21_aligned[i]
        bear_htf = close[i] < hma_12h_50_aligned[i]
        
        # === 1H LOCAL TREND (KAMA) ===
        kama_bullish = kama_slope[i] > 0 and close[i] > kama_1h[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama_1h[i]
        
        # === FISHER TRANSFORM SIGNALS (reversal entries) ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # crossing up
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # crossing down
        fisher_extreme_oversold = fisher[i] < -2.0
        fisher_extreme_overbought = fisher[i] > 2.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 0.8  # at least 80% of avg volume
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRIES (need HTF bull + local confirmation)
        if bull_htf or kama_bullish:
            # Fisher reversal + RSI confirmation
            if fisher_oversold and rsi_oversold and vol_confirmed:
                new_signal = LONG_SIZE
            # Extreme Fisher oversold (works without RSI)
            elif fisher_extreme_oversold and kama_bullish:
                new_signal = LONG_SIZE * 0.8
            # KAMA bullish + Fisher turning up
            elif kama_bullish and fisher[i] > fisher_signal[i] and fisher[i] < 0:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (need HTF bear + local confirmation)
        if bear_htf or kama_bearish:
            # Fisher reversal + RSI confirmation
            if fisher_overbought and rsi_overbought and vol_confirmed:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Extreme Fisher overbought (works without RSI)
            elif fisher_extreme_overbought and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # KAMA bearish + Fisher turning down
            elif kama_bearish and fisher[i] < fisher_signal[i] and fisher[i] > 0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: Fisher < -1.0 + KAMA bullish (simpler entry)
            if bull_htf and kama_bullish and fisher[i] < -1.0:
                new_signal = LONG_SIZE * 0.5
            # Short: Fisher > 1.0 + KAMA bearish (simpler entry)
            elif bear_htf and kama_bearish and fisher[i] > 1.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and fisher[i] > 2.0:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -2.0:
            new_signal = 0.0
        
        # Trend reversal exit (HTF regime flip)
        if in_position and position_side > 0 and bear_htf and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_htf and kama_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals