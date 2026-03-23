#!/usr/bin/env python3
"""
Experiment #687: 1d Primary + 1w HTF — Dual Regime with HMA/RSI/Donchian

Hypothesis: Daily timeframe with weekly HTF filter captures major trend moves while
avoiding noise. Key innovations based on proven 1d patterns:
1. HMA(21/50) crossover for trend direction — smoother than EMA, less lag
2. RSI(14) pullback entries in trend — buy dips in uptrend, sell rallies in downtrend
3. Choppiness Index regime — CHOP>55=range(mean-revert), CHOP<45=trend(breakout)
4. Donchian(20) breakout for trend confirmation
5. Weekly HMA for macro bias — only trade in direction of weekly trend
6. ATR(14) trailing stop at 2.5x for risk management
7. LOOSE entry thresholds to ensure 20-50 trades/year target

Why this should work where 1d strategies failed before:
- HMA crossover is proven on SOL (+0.879 Sharpe in research)
- Dual regime adapts to market conditions (trend vs range)
- 1d TF = ~20-50 trades/year (optimal for fee drag)
- Weekly HTF prevents counter-trend trades in major moves
- RSI pullback entries catch retracements in strong trends

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_donchian_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss_pad).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(high)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
        
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate and align HTF (1w) indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after warmup period for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_range_regime = chop_value > 55
        is_trend_regime = chop_value < 45
        
        # === WEEKLY MACRO BIAS (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND (HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_oversold = rsi_14[i] < 30
        rsi_extreme_overbought = rsi_14[i] > 70
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_weak = adx_14[i] < 20
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow ===
        if is_trend_regime:
            # Long: Weekly bullish + HMA bullish + (Donchian breakout OR RSI pullback)
            if weekly_bullish and hma_bullish:
                if donchian_breakout_long and adx_strong:
                    desired_signal = SIZE_LONG
                elif rsi_oversold and close[i] > hma_50[i]:
                    desired_signal = SIZE_LONG
            
            # Short: Weekly bearish + HMA bearish + (Donchian breakout OR RSI rally)
            elif weekly_bearish and hma_bearish:
                if donchian_breakout_short and adx_strong:
                    desired_signal = -SIZE_SHORT
                elif rsi_overbought and close[i] < hma_50[i]:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: RANGING (CHOP > 55) — Mean Reversion ===
        elif is_range_regime:
            # Long: RSI extreme oversold + price near Donchian lower + Weekly bullish bias
            if rsi_extreme_oversold:
                if close[i] < donchian_lower[i-1] * 1.02 or weekly_bullish:
                    desired_signal = SIZE_LONG * 0.5
            
            # Short: RSI extreme overbought + price near Donchian upper + Weekly bearish bias
            elif rsi_extreme_overbought:
                if close[i] > donchian_upper[i-1] * 0.98 or weekly_bearish:
                    desired_signal = -SIZE_SHORT * 0.5
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) — Mixed ===
        else:
            # Use HMA crossover with RSI confirmation
            if hma_bullish and rsi_oversold and weekly_bullish:
                desired_signal = SIZE_LONG * 0.5
            elif hma_bearish and rsi_overbought and weekly_bearish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if hma_bullish and rsi_14[i] < 75:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                if hma_bearish and rsi_14[i] > 25:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= SIZE_LONG:
                desired_signal = SIZE_LONG
            else:
                desired_signal = SIZE_LONG * 0.5
        elif desired_signal < 0:
            if desired_signal <= -SIZE_SHORT:
                desired_signal = -SIZE_SHORT
            else:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals