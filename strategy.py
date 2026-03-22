#!/usr/bin/env python3
"""
Experiment #416: 30m Supertrend + 4h HMA Trend + RSI Pullback + Volume Filter

Hypothesis: After 415 failed experiments, the key insight is that 30m timeframe
needs STRONG higher-timeframe filtering to avoid noise while still capturing
more opportunities than 4h/12h strategies. This strategy uses:

1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 4h HMA (bullish bias)
   - Short only when price < 4h HMA (bearish bias)
   - HMA is smoother than EMA, critical for 30m/4h alignment

2. 30m SUPERTREND(10, 3.0) for entry timing:
   - Supertrend flips provide clear entry/exit signals
   - Less lag than EMA crossover, more reliable than RSI alone
   - Proven indicator in crypto futures trading

3. 30m RSI(14) PULLBACK FILTER:
   - Long: RSI 40-55 in uptrend (pullback, not oversold)
   - Short: RSI 45-60 in downtrend (rally, not overbought)
   - Avoids entering at extremes in strong trends

4. 4h ADX(14) REGIME FILTER:
   - ADX > 20 = trending (allow Supertrend signals)
   - ADX < 20 = ranging (reduce position size by 50%)
   - Prevents whipsaw in choppy 4h conditions

5. VOLUME CONFIRMATION:
   - Taker buy volume > 20-bar MA for long entries
   - Taker sell volume > 20-bar MA for short entries
   - Confirms institutional participation

6. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

7. POSITION SIZING: 0.25 discrete (conservative for 30m volatility)
   - Max 25% capital per position
   - Reduces to 0.125 in ranging regime (ADX < 20)
   - Discrete levels minimize fee churn

Why 30m should work better than previous attempts:
- More trades than 4h/12h (~50-80/year vs 20-40)
- Strong 4h filter prevents noise entries
- Supertrend + RSI combo catches trend continuations
- Volume filter reduces false breakouts
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 normal, 0.125 ranging regime
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_vol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        # ADX is smoothed DX
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1  # Start short
        else:
            # Update upper/lower bands based on previous supertrend
            if supertrend[i-1] == upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            
            # Determine supertrend value
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1  # Long
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1  # Short
    
    return supertrend, direction

def calculate_volume_ma(taker_buy_volume, close, period=20):
    """Calculate volume moving average."""
    vol_ma = pd.Series(taker_buy_volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    vol_ma = calculate_volume_ma(taker_buy_vol, close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_NORMAL = 0.25
    SIZE_RANGING = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 4h REGIME DETECTION ===
        trending_regime = adx_4h_aligned[i] > 20
        ranging_regime = adx_4h_aligned[i] <= 20
        
        # === SUPERTREND SIGNAL ===
        supertrend_long = supertrend_dir[i] == 1
        supertrend_short = supertrend_dir[i] == -1
        
        # === RSI PULLBACK FILTER ===
        # In uptrend: look for RSI pullback to 40-55 (not oversold <30)
        rsi_pullback_long = 40 <= rsi[i] <= 55
        # In downtrend: look for RSI rally to 45-60 (not overbought >70)
        rsi_pullback_short = 45 <= rsi[i] <= 60
        
        # === VOLUME CONFIRMATION ===
        # For long: taker buy volume > MA (buying pressure)
        vol_confirm_long = taker_buy_vol[i] > vol_ma[i]
        # For short: taker buy volume < MA (selling pressure = low buy vol)
        vol_confirm_short = taker_buy_vol[i] < vol_ma[i]
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        current_size = SIZE_RANGING if ranging_regime else SIZE_NORMAL
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + Supertrend long + RSI pullback + Volume confirm
        if bull_trend_4h and supertrend_long and rsi_pullback_long and vol_confirm_long:
            new_signal = current_size
        
        # SHORT ENTRY: 4h bearish + Supertrend short + RSI pullback + Volume confirm
        elif bear_trend_4h and supertrend_short and rsi_pullback_short and vol_confirm_short:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === SUPERTREND FLIP EXIT ===
        # Exit long if Supertrend flips short
        if in_position and position_side > 0 and supertrend_short:
            new_signal = 0.0
        
        # Exit short if Supertrend flips long
        if in_position and position_side < 0 and supertrend_long:
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