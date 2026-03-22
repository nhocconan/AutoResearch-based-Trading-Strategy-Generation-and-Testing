#!/usr/bin/env python3
"""
Experiment #351: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Volatility Regime + RSI

Hypothesis: After 30+ failed experiments, the pattern is clear:
1. Choppiness Index filters = 0 trades (exp 339, 350) — AVOID
2. Connors RSI = 0 trades on 4h — TOO RESTRICTIVE
3. Simple Donchian + HMA + RSI generated trades but negative Sharpe (exp 349)
4. Key insight: VOLATILITY REGIME switching works better than choppiness

This strategy uses:
1. 1d HMA(21) for major trend direction (crypto trends last weeks)
2. 4h KAMA(10) for adaptive trend (less whipsaw than HMA/EMA in chop)
3. ATR ratio (ATR7/ATR30) for volatility regime detection
4. RSI(14) 40-60 range for entries (NOT extremes — generates 3x more trades)
5. Dual entry logic: trend breakout OR mean reversion based on vol regime
6. ATR(14) 2.5x trailing stop (cut losers, let winners run)
7. 1w HMA(21) as secondary HTF filter for major regime confirmation

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to volatility — less whipsaw than fixed EMA/HMA
- Volatility regime (ATR ratio) is simpler and more reliable than Choppiness
- RSI 40-60 range (not 20/80) generates signals in normal conditions
- Dual entry logic ensures trades in both trending and ranging markets
- 1w HTF adds major regime confirmation without over-filtering

Position sizing: 0.25-0.30 (discrete levels to minimize churn)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 4h (1 trade every 7-14 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_volregime_rsi_1d1w_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er_num = np.abs(close - np.roll(close, period))
    er_den = np.zeros(n)
    for i in range(period, n):
        er_den[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = er_num / (er_den + 1e-10)
    er[:period] = np.nan
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[:period] = np.nan
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
    n = period
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return donchian_upper, donchian_lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HTF indicators (super major trend)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, period=10)
    kama_20 = calculate_kama(close, period=20)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]):
            continue
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        high_vol_regime = atr_ratio > 1.3  # Vol spike = mean reversion
        low_vol_regime = atr_ratio < 0.9   # Low vol = trend follow
        normal_vol = not high_vol_regime and not low_vol_regime
        
        # === 1D MAJOR TREND REGIME ===
        regime_bull_1d = close[i] > hma_1d_aligned[i]
        regime_bear_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W SUPER MAJOR TREND ===
        regime_bull_1w = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        regime_bear_1w = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # === 4H LOCAL TREND (KAMA) ===
        kama_bullish = kama_10[i] > kama_20[i]
        kama_bearish = kama_10[i] < kama_20[i]
        
        # KAMA slope
        kama_slope_up = kama_10[i] > kama_10[i-2] if i >= 2 else False
        kama_slope_down = kama_10[i] < kama_10[i-2] if i >= 2 else False
        
        # === RSI SIGNALS (40-60 range for more trades) ===
        rsi_neutral = 40.0 < rsi_14[i] < 60.0
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === PRICE POSITION ===
        price_above_bb_mid = close[i] > bb_mid[i] if not np.isnan(bb_mid[i]) else True
        price_below_bb_mid = close[i] < bb_mid[i] if not np.isnan(bb_mid[i]) else False
        near_bb_lower = close[i] < bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] > bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLER — fewer AND conditions) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime_bull_1d or regime_bull_1w:
            # Trend follow (low vol): Donchian breakout + KAMA bullish
            if low_vol_regime and donchian_breakout_long and kama_bullish:
                new_signal = LONG_STRONG
            
            # Trend follow (normal vol): KAMA bullish + RSI rising + price above BB mid
            elif normal_vol and kama_bullish and rsi_rising and price_above_bb_mid:
                new_signal = LONG_BASE
            
            # Mean reversion (high vol): RSI oversold + near BB lower
            elif high_vol_regime and rsi_oversold and near_bb_lower:
                new_signal = LONG_BASE
            
            # Simple: KAMA bullish + RSI neutral + 1d bull
            elif kama_bullish and rsi_neutral and regime_bull_1d:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES
        if regime_bear_1d or regime_bear_1w:
            # Trend follow (low vol): Donchian breakout + KAMA bearish
            if low_vol_regime and donchian_breakout_short and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # Trend follow (normal vol): KAMA bearish + RSI falling + price below BB mid
            elif normal_vol and kama_bearish and rsi_falling and price_below_bb_mid:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Mean reversion (high vol): RSI overbought + near BB upper
            elif high_vol_regime and rsi_overbought and near_bb_upper:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Simple: KAMA bearish + RSI neutral + 1d bear
            elif kama_bearish and rsi_neutral and regime_bear_1d:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year) ===
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if regime_bull_1d and kama_bullish and rsi_14[i] > 42.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear_1d and kama_bearish and rsi_14[i] < 58.0:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 65.0:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 35.0:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear_1d and kama_bearish:
                regime_reversal = True
            if position_side < 0 and regime_bull_1d and kama_bullish:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals