#!/usr/bin/env python3
"""
Experiment #451: 4h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + RSI Pullback

Hypothesis: After analyzing 450+ failed experiments, clear patterns emerge:
1. Donchian breakout + HMA trend worked on SOL (Sharpe +0.782 in research notes)
2. Pure trend strategies fail on BTC/ETH in bear markets (2022 crash, 2025 bear)
3. Need REGIME ADAPTIVE: trend-follow in trending markets, mean-revert in ranges
4. 4h TF with 1d HTF gives good balance: 20-50 trades/year, lower fee drag than 1h
5. Simpler entry logic = more trades (critical: need >=30 trades/symbol on train)

Key innovations vs failed strategies:
- Donchian(20) breakout for trend entry (proven on SOL)
- 1d HMA(21) for major trend bias (not over-filtering like 1w)
- RSI(14) pullback for entry timing (better than CRSI complexity)
- ATR volatility filter to avoid choppy whipsaw periods
- Asymmetric sizing: 0.30 long, 0.25 short (protects in bear markets)
- Relaxed entry thresholds to ensure >=30 trades/symbol

Why this might beat current best (Sharpe=0.435):
- Donchian breakouts catch major moves without over-trading
- 1d HTF trend filter avoids counter-trend trades in strong trends
- RSI pullback entries improve win rate vs pure breakout
- ATR filter reduces trades in low-vol chop (fee savings)
- 4h TF = ~30-50 trades/year = optimal fee/edge balance

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_regime_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    atr = tr_s.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volatility_ratio(atr_short, atr_long):
    """Calculate volatility ratio (ATR short / ATR long)."""
    return atr_short / (atr_long + 1e-10)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_50 = calculate_atr(high, low, close, 50)
    
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    adx_14 = calculate_adx(high, low, close, period=14)
    
    sma_200 = calculate_sma(close, period=200)
    
    # Volatility ratio for regime detection
    vol_ratio = calculate_volatility_ratio(atr_7, atr_50)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull bias (favor longs)
        # Price below 1d HMA = bear bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === VOLATILITY REGIME ===
        # vol_ratio > 1.5 = high vol (trend follow)
        # vol_ratio < 0.8 = low vol (mean revert / avoid trading)
        high_vol = vol_ratio[i] > 1.3
        low_vol = vol_ratio[i] < 0.9
        
        # === ADX TREND STRENGTH ===
        # ADX > 25 = trending market
        # ADX < 20 = ranging market
        is_trending = adx_14[i] > 22.0  # relaxed for more trades
        is_ranging = adx_14[i] < 25.0
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper = bullish momentum
        # Breakout below lower = bearish momentum
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995  # near breakout
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005  # near breakout
        
        # === RSI PULLBACK SIGNALS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_neutral = 40.0 < rsi_14[i] < 60.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if bull_regime or above_sma200:
            # Trending market: Donchian breakout + HMA confirmation
            if is_trending and hma_bullish and donchian_breakout_long:
                new_signal = LONG_SIZE
            # Trending market: RSI pullback to support in uptrend
            elif is_trending and hma_bullish and rsi_oversold and rsi_neutral:
                new_signal = LONG_SIZE
            # Ranging market: RSI extreme oversold mean reversion
            elif is_ranging and rsi_extreme_oversold and hma_bullish:
                new_signal = LONG_SIZE
            # High vol breakout: momentum entry
            elif high_vol and donchian_breakout_long and hma_bullish:
                new_signal = LONG_SIZE
            # General HMA bullish + RSI confirmation (simpler entry for more trades)
            elif hma_bullish and rsi_14[i] < 50.0 and not rsi_overbought:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES
        if bear_regime or below_sma200:
            # Trending market: Donchian breakdown + HMA confirmation
            if is_trending and hma_bearish and donchian_breakout_short:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trending market: RSI pullback to resistance in downtrend
            elif is_trending and hma_bearish and rsi_overbought and rsi_neutral:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Ranging market: RSI extreme overbought mean reversion
            elif is_ranging and rsi_extreme_overbought and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # High vol breakdown: momentum entry
            elif high_vol and donchian_breakout_short and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # General HMA bearish + RSI confirmation (simpler entry for more trades)
            elif hma_bearish and rsi_14[i] > 50.0 and not rsi_oversold:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: HMA bullish + RSI < 45 (simple momentum)
            if bull_regime and hma_bullish and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.5
            # Short: HMA bearish + RSI > 55 (simple momentum)
            elif bear_regime and hma_bearish and rsi_14[i] > 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === AVOID LOW VOLATILITY (fee savings) ===
        if low_vol and new_signal != 0.0:
            # Reduce position size in low vol (less conviction)
            new_signal = new_signal * 0.6
        
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
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
            new_signal = 0.0
        
        # Donchian opposite breakout (trend reversal)
        if in_position and position_side > 0 and donchian_breakout_short:
            new_signal = 0.0
        if in_position and position_side < 0 and donchian_breakout_long:
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