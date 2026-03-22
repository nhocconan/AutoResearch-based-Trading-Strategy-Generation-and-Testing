#!/usr/bin/env python3
"""
Experiment #511: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: After 499 failed experiments (mostly volspike/CRSI/Choppiness combos), try a 
SIMPLER approach based on KAMA (Kaufman Adaptive Moving Average) which naturally adapts 
to market noise without complex regime switching.

Key insights from failures:
- 448+ volspike/CRSI/Choppiness strategies FAILED — need DIFFERENT approach
- Many strategies got Sharpe=0.000 (ZERO trades) — conditions TOO STRICT
- KAMA adapts ER (Efficiency Ratio): fast in trends, slow in chop
- ADX > 20 confirms trend, ADX < 20 = range (simpler than Choppiness Index)
- RSI pullback entries work better than breakouts in crypto

Why this might beat current best (Sharpe=0.435):
- KAMA is DIFFERENT from HMA/EMA (not yet exhausted in 499 experiments)
- Simpler entry logic = MORE trades (critical: need >=30/symbol on train)
- ADX regime filter prevents whipsaw without over-complicating
- 4h TF targets 25-50 trades/year (optimal fee/trade balance)
- Asymmetric sizing: 0.30 long, 0.25 short (protects in bear markets)

Position sizing: 0.25-0.30 (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-60 trades/symbol on train, >=5 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_regime_1d_v1"
timeframe = "4h"
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
    Adapts to market noise via Efficiency Ratio (ER).
    ER near 1 = trending (fast KAMA), ER near 0 = choppy (slow KAMA)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Price change over er_period
    price_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of absolute price movements (volatility)
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = price_change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation (iterative for proper adaptation)
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]  # Start with price
    
    for i in range(er_period + 1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    plus_di = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI indicators
    plus_di = 100 * (plus_di / (atr + 1e-10))
    minus_di = 100 * (minus_di / (atr + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    kama_1d_20 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    sma_1d_50 = calculate_sma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA for adaptive trend following
    kama_4h_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    # Recalculate kama_50 with different slow period
    kama_4h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    
    # ADX for regime detection
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # RSI for entry timing
    rsi_14 = calculate_rsi(close, 14)
    
    # SMA for additional trend filter
    sma_4h_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_1d_20_aligned[i]) or np.isnan(sma_1d_50_aligned[i]):
            continue
        if np.isnan(kama_4h_20[i]) or np.isnan(kama_4h_50[i]):
            continue
        if np.isnan(adx[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > kama_1d_20_aligned[i]
        bear_regime = close[i] < kama_1d_20_aligned[i]
        
        # === 4H TREND (KAMA crossover) ===
        kama_bull = kama_4h_20[i] > kama_4h_50[i]
        kama_bear = kama_4h_20[i] < kama_4h_50[i]
        
        # KAMA slope (trending up/down)
        kama_slope_up = kama_4h_20[i] > kama_4h_20[i-1] if i > 0 else False
        kama_slope_down = kama_4h_20[i] < kama_4h_20[i-1] if i > 0 else False
        
        # === ADX REGIME DETECTION ===
        trending = adx[i] > 20.0  # ADX > 20 = trending market
        ranging = adx[i] <= 20.0  # ADX <= 20 = ranging/choppy
        
        # DI crossover for direction
        di_bull = plus_di[i] > minus_di[i]
        di_bear = plus_di[i] < minus_di[i]
        
        # === RSI ENTRY TIMING ===
        rsi_oversold = rsi_14[i] < 40.0  # Relaxed for more trades
        rsi_overbought = rsi_14[i] > 60.0  # Relaxed for more trades
        rsi_extreme_low = rsi_14[i] < 30.0
        rsi_extreme_high = rsi_14[i] > 70.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_4h_200[i] if not np.isnan(sma_4h_200[i]) else True
        below_sma200 = close[i] < sma_4h_200[i] if not np.isnan(sma_4h_200[i]) else False
        
        # === ENTRY LOGIC — KAMA ADAPTIVE + ADX REGIME + RSI TIMING ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for frequency)
        # Condition 1: Bull regime + KAMA bull + RSI pullback (trend continuation)
        if bull_regime and kama_bull and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 2: Trending + DI bull + RSI not overbought (momentum)
        elif trending and di_bull and rsi_14[i] < 65.0:
            new_signal = LONG_SIZE * 0.8
        # Condition 3: RSI extreme low + above SMA200 (oversold in uptrend)
        elif rsi_extreme_low and above_sma200:
            new_signal = LONG_SIZE
        # Condition 4: KAMA crossover bull + ADX rising (trend start)
        elif kama_bull and kama_slope_up and adx[i] > adx[i-5] if i > 5 else False:
            new_signal = LONG_SIZE * 0.7
        # Condition 5: Simple KAMA bull + RSI neutral (baseline long)
        elif kama_bull and 35.0 < rsi_14[i] < 55.0:
            new_signal = LONG_SIZE * 0.6
        # Condition 6: 1d bull + 4h KAMA bull (HTF confluence)
        elif bull_regime and kama_bull and di_bull:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + KAMA bear + RSI bounce (trend continuation)
            if bear_regime and kama_bear and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Trending + DI bear + RSI not oversold (momentum)
            elif trending and di_bear and rsi_14[i] > 35.0:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 3: RSI extreme high + below SMA200 (overbought in downtrend)
            elif rsi_extreme_high and below_sma200:
                new_signal = -SHORT_SIZE
            # Condition 4: KAMA crossover bear + ADX rising (trend start)
            elif kama_bear and kama_slope_down and adx[i] > adx[i-5] if i > 5 else False:
                new_signal = -SHORT_SIZE * 0.7
            # Condition 5: Simple KAMA bear + RSI neutral (baseline short)
            elif kama_bear and 45.0 < rsi_14[i] < 65.0:
                new_signal = -SHORT_SIZE * 0.6
            # Condition 6: 1d bear + 4h KAMA bear (HTF confluence)
            elif bear_regime and kama_bear and di_bear:
                new_signal = -SHORT_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on regime flip or RSI extreme high
        if in_position and position_side > 0:
            if bear_regime and kama_bear:
                new_signal = 0.0
            elif rsi_extreme_high:
                new_signal = 0.0
        
        # Exit short on regime flip or RSI extreme low
        if in_position and position_side < 0:
            if bull_regime and kama_bull:
                new_signal = 0.0
            elif rsi_extreme_low:
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
                # Flip position
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